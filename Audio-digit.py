import os, re, glob, json, math, random
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "recordings"      # path to FSDD recordings
ARTIFACTS = "fsdd_cnn_mel.pt"  # weights + metadata (torch.save dict)

# Audio / Feature
SR = None                   # keep native (FSDD is 8 kHz)
N_MELS = 40
N_FFT = 512                 # stable mel filters for 8kHz
HOP_LEN = 80                # ~10 ms @ 8k
WIN_LEN = 200               # ~25 ms @ 8k
FMIN = 20
FMAX = None                 # default -> sr/2 per file
TARGET_RMS_DBFS = -20       # RMS normalization target

# Train
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Augment (train-time)
NOISE_PROB = 0.30           # apply white noise to 30% of training samples
NOISE_LEVEL = 0.005         # std of white noise

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------
# Filename parsing
# ----------------------------
NAME_RE = re.compile(r"(?P<digit>\d)_(?P<speaker>[A-Za-z]+)_\d+\.wav")

def parse_digit_and_speaker(path):
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f"Unexpected filename: {path}")
    return int(m.group("digit")), m.group("speaker")

# ----------------------------
# Audio utils
# ----------------------------
def rms_dbfs(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

def rms_normalize(x: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    cur = rms_dbfs(x)
    gain_db = target_dbfs - cur
    gain = 10.0 ** (gain_db / 20.0)
    y = np.clip(x * gain, -1.0, 1.0)
    return y

def add_white_noise(x: np.ndarray, noise_level: float = NOISE_LEVEL) -> np.ndarray:
    """Add white Gaussian noise to waveform and clip to [-1,1]."""
    noise = np.random.randn(x.shape[0]) * noise_level
    y = x + noise
    return np.clip(y, -1.0, 1.0)

def time_shift(x: np.ndarray, sr: int, shift_ms: float) -> np.ndarray:
    """Shift waveform by +shift_ms (right) with zero padding (no wrap)."""
    if shift_ms == 0:
        return x
    samples = int(abs(shift_ms) * sr / 1000.0)
    if samples == 0: 
        return x
    if shift_ms > 0:
        # delay: pad left
        y = np.concatenate([np.zeros(samples, dtype=x.dtype), x])[: len(x)]
    else:
        # advance: pad right
        y = np.concatenate([x, np.zeros(samples, dtype=x.dtype)])[samples : samples + len(x)]
    return y

def logmel(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX if FMAX else sr/2, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # shape (n_mels, T)
    return S_db.astype(np.float32)

def pad_or_trunc(x: np.ndarray, target_T: int) -> np.ndarray:
    # x: (C, T). Zero-pad or center-truncate to target_T
    C, T = x.shape
    if T == target_T: return x
    if T < target_T:
        pad = target_T - T
        left = pad // 2
        right = pad - left
        out = np.pad(x, ((0, 0), (left, right)), mode="constant", constant_values=0.0)
        return out
    # truncate
    start = (T - target_T) // 2
    end = start + target_T
    return x[:, start:end]

# ----------------------------
# Dataset preparation
# ----------------------------
def list_wavs(data_dir):
    # recursive + case-insensitive
    pats = [
        os.path.join(data_dir, "**", "*.wav"),
        os.path.join(data_dir, "**", "*.WAV"),
    ]
    files = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(files)

def scan_dataset(data_dir):
    """Return paths, labels, speakers without loading features."""
    paths = list_wavs(data_dir)
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No WAV files found under '{data_dir}'. "
            "Tip: check the path, ensure files end with .wav/.WAV, "
            "or that your files aren’t nested in a different folder."
        )
    y_list, spk_list = [], []
    bad = 0
    for p in paths:
        try:
            digit, spk = parse_digit_and_speaker(p)
        except ValueError:
            bad += 1
            continue
        y_list.append(digit)
        spk_list.append(spk)
    if len(y_list) == 0:
        raise ValueError(
            "WAV files were found, but none matched the expected FSDD filename pattern "
            "'<digit>_<speaker>_<index>.wav' (e.g., 0_jackson_0.wav)."
        )
    if bad:
        print(f"Warning: skipped {bad} file(s) with unexpected names.")
    return np.array(paths), np.array(y_list, dtype=np.int64), np.array(spk_list)

def compute_target_T(example_path: str) -> int:
    """Get a stable target T from a reference example (no augmentation)."""
    wav, sr = librosa.load(example_path, sr=SR, mono=True)
    wav = rms_normalize(wav, TARGET_RMS_DBFS)
    feat = logmel(wav, sr)  # (N_MELS, T)
    return feat.shape[1]

def extract_subset(paths_subset, target_T: int, transform=None):
    """
    transform: callable(wav: np.ndarray, sr: int) -> np.ndarray
               applied after RMS normalize
    """
    X_list = []
    for p in paths_subset:
        wav, sr = librosa.load(p, sr=SR, mono=True)
        wav = rms_normalize(wav, TARGET_RMS_DBFS)
        if transform is not None:
            wav = transform(wav, sr)
        feat = logmel(wav, sr)          # (N_MELS, T)
        feat = pad_or_trunc(feat, target_T)
        X_list.append(feat)
    X = np.stack(X_list, axis=0)    # (N, C, T)
    return X

def compute_channel_stats(X_train: np.ndarray):
    # X_train: (N, C, T). Compute per-channel mean/std over all frames.
    N, C, T = X_train.shape
    sum_c = X_train.sum(axis=(0, 2))          # (C,)
    sumsq_c = (X_train ** 2).sum(axis=(0, 2)) # (C,)
    count = N * T
    mean = sum_c / count
    var = (sumsq_c / count) - (mean ** 2)
    std = np.sqrt(np.clip(var, 1e-8, None))
    return mean.astype(np.float32), std.astype(np.float32)

def apply_channel_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    # X: (N, C, T)
    return ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)

class MelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)   # (N, C, T)
        self.y = torch.from_numpy(y)   # (N,)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------
# Tiny 1D-CNN (depthwise-separable)
# ----------------------------
class DSConv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, p=0.1):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm1d(in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x = self.dw_bn(self.dw(x))
        x = self.act(x)
        x = self.pw_bn(self.pw(x))
        x = self.act(x)
        x = self.drop(x)
        return x

class TinyMelCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=10):
        super().__init__()
        self.block1 = DSConv1DBlock(n_mels, 64, k=5, p=0.1)
        self.block2 = DSConv1DBlock(64, 64, k=5, p=0.1)
        self.head = nn.Linear(64, n_classes)
        # init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    def forward(self, x):           # x: (B, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=-1)          # global average pool over time -> (B, 64)
        logits = self.head(x)
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------
# Train / Eval helpers
# ----------------------------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(yb.numpy().tolist())
    return np.array(y_true), np.array(y_pred)

# ----------------------------
# Training
# ----------------------------
def run_train(data_dir=DATA_DIR):
    print("Scanning dataset…")
    paths, y_all, spk_all = scan_dataset(data_dir)
    print(f"Found {len(paths)} files. Example: {paths[0]}")

    # Stratified split by label (digits are balanced)
    idx = np.arange(len(y_all))
    tr_idx, tmp_idx, y_tr, y_tmp = train_test_split(idx, y_all, test_size=0.30, stratify=y_all, random_state=SEED)
    va_idx, te_idx, y_va, y_te = train_test_split(tmp_idx, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    # Establish a common target_T from a reference file (no aug)
    target_T = compute_target_T(paths[0])

    # Extract features; apply noise ONLY to training subset
    print("Featurizing (with white noise on train)…")
    X_tr_raw = extract_subset(
        paths[tr_idx], target_T,
        transform=lambda w, sr: add_white_noise(w, NOISE_LEVEL) if (random.random() < NOISE_PROB) else w
    )
    X_va_raw = extract_subset(paths[va_idx], target_T, transform=None)
    X_te_raw = extract_subset(paths[te_idx], target_T, transform=None)

    # Channel stats on TRAIN only
    mean_c, std_c = compute_channel_stats(X_tr_raw)
    X_tr = apply_channel_standardize(X_tr_raw, mean_c, std_c)
    X_va = apply_channel_standardize(X_va_raw, mean_c, std_c)
    X_te = apply_channel_standardize(X_te_raw, mean_c, std_c)

    train_ds = MelDataset(X_tr, y_tr)
    val_ds   = MelDataset(X_va, y_va)
    test_ds  = MelDataset(X_te, y_te)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = TinyMelCNN(n_mels=N_MELS, n_classes=10).to(DEVICE)
    print(f"Model params: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_va = 0.0
    best_state = None
    patience, bad = 5, 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_ld, opt, loss_fn)
        va_loss, va_acc = eval_epoch(model, val_ld, loss_fn)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Load best and test
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = eval_epoch(model, test_ld, loss_fn)
    print(f"Test acc (clean): {te_acc:.4f}")

    # Confusion matrix + classification report on clean test
    y_true, y_pred = predict_all(model, test_ld)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=list(range(10)), digits=4))

    # Save artifacts
    artifacts = {
        "state_dict": model.state_dict(),
        "mel_cfg": {
            "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LEN,
            "win_length": WIN_LEN, "fmin": FMIN, "fmax": FMAX
        },
        "target_T": int(target_T),
        "channel_mean": mean_c,
        "channel_std": std_c,
        "rms_target_dbfs": TARGET_RMS_DBFS,
        "label_map": list(range(10)),
        "split_seed": SEED,  # keep for reproducible robust eval
    }
    torch.save(artifacts, ARTIFACTS)
    print(f"Saved -> {ARTIFACTS}")

# ----------------------------
# Robustness evaluation
# ----------------------------
@torch.no_grad()
def run_robust_eval(data_dir=DATA_DIR, artifacts_path=ARTIFACTS,
                    noise_levels=(0.002, 0.005, 0.01, 0.02),
                    shifts_ms=(0, 10, 20, 40)):
    print("Loading artifacts…")
    bundle = torch.load(artifacts_path, map_location="cpu")
    state_dict = bundle["state_dict"]
    mel_cfg = bundle["mel_cfg"]
    target_T = bundle["target_T"]
    mean_c = bundle["channel_mean"]
    std_c = bundle["channel_std"]
    rms_target = bundle["rms_target_dbfs"]
    split_seed = bundle.get("split_seed", SEED)

    # Model
    model = TinyMelCNN(n_mels=mel_cfg["n_mels"], n_classes=10).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Rebuild the *same* split with the saved seed
    paths, y_all, _ = scan_dataset(data_dir)
    idx = np.arange(len(y_all))
    tr_idx, tmp_idx, _, y_tmp = train_test_split(idx, y_all, test_size=0.30, stratify=y_all, random_state=split_seed)
    va_idx, te_idx, _, y_te = train_test_split(tmp_idx, y_tmp, test_size=0.50, stratify=y_tmp, random_state=split_seed)
    test_paths = paths[te_idx]

    def _standardize_and_wrap(X_raw):
        return apply_channel_standardize(X_raw, mean_c, std_c)

    # Helper to make a loader from a transform
    def _make_test_loader(transform):
        X_raw = extract_subset(
            test_paths, target_T,
            transform=lambda w, sr: transform(rms_normalize(w, rms_target), sr)
        )
        X = _standardize_and_wrap(X_raw)
        ds = MelDataset(X, y_te)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Baseline (clean)
    print("\n[Robustness] Clean test")
    dl_clean = _make_test_loader(lambda w, sr: w)  # identity
    clean_acc = eval_epoch(model, dl_clean, nn.CrossEntropyLoss())[1]
    print(f"Accuracy: {clean_acc:.4f}")

    # White noise sweep
    print("\n[Robustness] White noise sweep")
    for nl in noise_levels:
        dl = _make_test_loader(lambda w, sr, _nl=nl: add_white_noise(w, _nl))
        acc = eval_epoch(model, dl, nn.CrossEntropyLoss())[1]
        print(f"noise_std={nl:.4f} -> acc={acc:.4f}")

    # Time-shift sweep
    print("\n[Robustness] Time shift sweep")
    for ms in shifts_ms:
        dl = _make_test_loader(lambda w, sr, _ms=ms: time_shift(w, sr, _ms))
        acc = eval_epoch(model, dl, nn.CrossEntropyLoss())[1]
        print(f"shift_ms={ms:>3} -> acc={acc:.4f}")

# ----------------------------
# Single-file inference
# ----------------------------
@torch.no_grad()
def infer_one(wav_path, artifacts_path=ARTIFACTS):
    print("Loading model…")
    bundle = torch.load(artifacts_path, map_location="cpu")
    state_dict = bundle["state_dict"]
    mel_cfg = bundle["mel_cfg"]
    target_T = bundle["target_T"]
    mean_c = bundle["channel_mean"]
    std_c = bundle["channel_std"]
    rms_target = bundle["rms_target_dbfs"]

    model = TinyMelCNN(n_mels=mel_cfg["n_mels"], n_classes=10)
    model.load_state_dict(state_dict)
    model.eval()

    # Load + features (no augmentation at inference)
    y_wav, sr = librosa.load(wav_path, sr=SR, mono=True)
    y_wav = rms_normalize(y_wav, rms_target)
    S = librosa.feature.melspectrogram(
        y=y_wav, sr=sr, n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg["win_length"], n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"] if mel_cfg["fmax"] else sr/2, power=2.0
    )
    X = librosa.power_to_db(S, ref=np.max).astype(np.float32)  # (C, T)
    X = pad_or_trunc(X, target_T)
    X = ((X - mean_c[None, :]) / std_c[None, :]).astype(np.float32)
    X = torch.from_numpy(X).unsqueeze(0)  # (1, C, T)
    logits = model(X)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    print(f"Predicted: {pred} | probs: {np.round(probs, 4)}")
    return pred, probs

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FSDD lightweight CNN baseline (+ white noise + robustness)")
    sub = parser.add_subparsers(dest="cmd")

    tr = sub.add_parser("train", help="Train tiny 1D-CNN on FSDD")
    tr.add_argument("--data_dir", type=str, default=DATA_DIR)

    inf = sub.add_parser("infer", help="Run inference on a single WAV")
    inf.add_argument("wav_path", type=str, help="Path to a WAV file")
    inf.add_argument("--artifacts", type=str, default=ARTIFACTS)

    rob = sub.add_parser("robust", help="Robustness evaluation (noise + time shift) on test split")
    rob.add_argument("--data_dir", type=str, default=DATA_DIR)
    rob.add_argument("--artifacts", type=str, default=ARTIFACTS)
    rob.add_argument("--noise_levels", type=float, nargs="*", default=[0.002, 0.005, 0.01, 0.02])
    rob.add_argument("--shifts_ms", type=float, nargs="*", default=[0, 10, 20, 40])

    args = parser.parse_args()
    if args.cmd == "train":
        run_train(args.data_dir)
    elif args.cmd == "infer":
        infer_one(args.wav_path, args.artifacts)
    elif args.cmd == "robust":
        run_robust_eval(
            data_dir=args.data_dir,
            artifacts_path=args.artifacts,
            noise_levels=tuple(args.noise_levels),
            shifts_ms=tuple(args.shifts_ms),
        )
    else:
        parser.print_help()
