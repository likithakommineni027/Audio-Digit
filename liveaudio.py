import argparse
import sys
import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa

# ----------------------------
# Minimal helpers (must match training)
# ----------------------------

def rms_dbfs(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

def rms_normalize(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    cur = rms_dbfs(x)
    gain_db = target_dbfs - cur
    gain = 10.0 ** (gain_db / 20.0)
    y = np.clip(x * gain, -1.0, 1.0)
    return y.astype(np.float32)

def pad_or_trunc(x: np.ndarray, target_T: int) -> np.ndarray:
    # x: (C, T)
    C, T = x.shape
    if T == target_T:
        return x
    if T < target_T:
        pad = target_T - T
        left = pad // 2
        right = pad - left
        return np.pad(x, ((0, 0), (left, right)), mode="constant", constant_values=0.0)
    start = (T - target_T) // 2
    end = start + target_T
    return x[:, start:end]

# ----------------------------
# Tiny 1D‑CNN (depthwise‑separable) – identical to training
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
    def __init__(self, n_mels=40, n_classes=10):
        super().__init__()
        self.block1 = DSConv1DBlock(n_mels, 64, k=5, p=0.1)
        self.block2 = DSConv1DBlock(64, 64, k=5, p=0.1)
        self.head = nn.Linear(64, n_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    def forward(self, x):  # x: (B, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=-1)
        return self.head(x)

# ----------------------------
# Core logic
# ----------------------------

def load_bundle(artifacts_path: str, device: torch.device):
    bundle = torch.load(artifacts_path, map_location="cpu")
    state_dict = bundle["state_dict"]
    mel_cfg = bundle["mel_cfg"]
    target_T = int(bundle["target_T"])  # frames
    mean_c = np.array(bundle["channel_mean"], dtype=np.float32)  # (C,)
    std_c = np.array(bundle["channel_std"], dtype=np.float32)    # (C,)
    rms_target = float(bundle["rms_target_dbfs"])                # e.g., -20

    model = TinyMelCNN(n_mels=mel_cfg["n_mels"], n_classes=10)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model, mel_cfg, target_T, mean_c, std_c, rms_target


def mel_from_audio(y: np.ndarray, sr: int, mel_cfg: dict) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg["win_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"],
        fmax=(mel_cfg["fmax"] if mel_cfg["fmax"] else sr / 2),
        power=2.0,
    )
    X = librosa.power_to_db(S, ref=np.max).astype(np.float32)  # (C, T)
    return X


def infer_once_window(model, device, y_window, sr, mel_cfg, target_T, mean_c, std_c):
    # Normalize to target loudness first (same as training)
    y_norm = rms_normalize(y_window, target_dbfs=rms_target_dbfs)
    X = mel_from_audio(y_norm, sr, mel_cfg)            # (C, T)
    X = pad_or_trunc(X, target_T)                      # (C, target_T)
    # Per‑channel standardization; broadcast along time
    X = (X - mean_c[:, None]) / std_c[:, None]
    X = torch.from_numpy(X).unsqueeze(0).to(device)     # (1, C, T)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs


# Globals filled in main (used inside infer_once_window)
rms_target_dbfs = -20.0  # overwritten


def run_stream(args):
    global rms_target_dbfs

    # Decide device explicitly (fixing 'auto' string issue)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, mel_cfg, target_T, mean_c, std_c, rms_target_dbfs = load_bundle(args.artifacts, device)

    # Compute exact samples needed to yield `target_T` mel frames
    hop = int(mel_cfg["hop_length"])  # samples
    win = int(mel_cfg["win_length"])  # samples
    needed_samples = (target_T - 1) * hop + win

    # Rolling buffer + step size (how often we run inference)
    step_samples = int(args.step_sec * args.sr)
    buf = deque(maxlen=needed_samples)

    # Simple EMA smoother over probabilities
    ema_probs = None
    alpha = float(args.ema)  # 0..1 (higher = smoother)

    # Voice activity threshold to avoid spamming during silence
    vad_thresh_dbfs = float(args.vad_dbfs)

    print("Loaded model. Listening on mic… Press Ctrl+C to stop.")
    print(f"Device={device} | sr={args.sr} Hz | window={needed_samples/args.sr:.2f}s | step={args.step_sec:.2f}s | target_T={target_T}")

    def audio_cb(indata, frames, time_info, status):
        if status:
            # Over/underflows etc.
            print(status, file=sys.stderr)
        # indata: (frames, channels), float32 in [-1,1]
        mono = indata[:, 0].copy()
        buf.extend(mono.tolist())

    with sd.InputStream(samplerate=args.sr, channels=1, dtype='float32', callback=audio_cb, blocksize=0):
        last_infer_t = time.time()
        while True:
            try:
                now = time.time()
                if now - last_infer_t < args.step_sec:
                    time.sleep(0.005)
                    continue
                last_infer_t = now

                if len(buf) < buf.maxlen:
                    continue  # not enough audio yet

                y = np.asarray(buf, dtype=np.float32)

                # Skip if silence
                if rms_dbfs(y) < vad_thresh_dbfs:
                    continue

                probs = infer_once_window(model, device, y, args.sr, mel_cfg, target_T, mean_c, std_c)

                # EMA smoothing (optional)
                if ema_probs is None:
                    ema_probs = probs
                else:
                    ema_probs = alpha * ema_probs + (1 - alpha) * probs

                p = ema_probs if args.ema > 0 else probs
                topk_idx = p.argsort()[-args.topk:][::-1]
                pred = int(p.argmax())

                if args.show_probs:
                    msg = " | ".join([f"{i}:{p[i]:.2f}" for i in topk_idx])
                    print(f"pred={pred}  [{msg}]")
                else:
                    print(f"pred={pred}  conf={p[pred]:.2f}")

            except KeyboardInterrupt:
                print("\nStopping…")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live mic inference for FSDD tiny CNN")
    parser.add_argument("--artifacts", type=str, default="fsdd_cnn_mel.pt", help="Path to saved artifacts .pt")
    parser.add_argument("--sr", type=int, default=8000, help="Microphone sample rate (match training; FSDD is 8kHz)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--topk", type=int, default=3, help="Show top‑k labels")
    parser.add_argument("--ema", type=float, default=0.6, help="EMA smoothing factor in [0,1]; 0 disables smoothing")
    parser.add_argument("--vad_dbfs", type=float, default=-45.0, help="Skip frames quieter than this dBFS")
    parser.add_argument("--step_sec", type=float, default=0.10, help="How often to run inference (seconds)")
    parser.add_argument("--show_probs", action="store_true", help="Print top‑k probs instead of only top‑1")

    args = parser.parse_args()
    run_stream(args)
