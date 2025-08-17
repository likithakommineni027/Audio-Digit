FSDD Lightweight Audio Digit Classifier
A tiny, fast baseline that turns spoken digits (0–9) into predictions using log-mel spectrograms and a depthwise-separable 1D-CNN. It trains in minutes, stays small (<100k params), and includes a built-in robustness suite (white noise + time shift).
What this really means is: audio in → digit out, with clean preprocessing, reproducible splits, and one command per task.
1) Approach
Data & filenames
We use the Free Spoken Digit Dataset (FSDD). Files are expected to follow digit_speaker_index.wav (e.g., 3_jackson_12.wav). The script validates names and skips unknown patterns.
Preprocessing
Load WAV at native rate (FSDD is 8 kHz).
RMS normalize each clip to a target loudness (−20 dBFS) for consistent gain.
Compute log-mel spectrograms: 40 mels, n_fft=512, hop=80, win=200, fmin=20.
Center pad/truncate to a shared temporal length T computed from a reference file.
Compute per-channel (per-mel) mean/std on the train split only and standardize train/val/test with those stats.
Model
A compact depthwise-separable 1D-CNN with ReLU, BN, dropout, and global average pooling over time, followed by a linear head → 10 classes. Depthwise + pointwise keeps parameters small while modeling time context.
Training
Stratified train/val/test split with a fixed seed for reproducibility.
AdamW optimizer, cross-entropy loss, early stopping on validation accuracy.
Lightweight on-the-fly white-noise augmentation applied to a probability of train clips (defaults: 30% at σ=0.005). Augment after RMS normalization.
Evaluation & Robustness
Report accuracy, confusion matrix, and classification report on the clean test set.
Robustness sweeps: accuracy under varying white-noise levels and time shifts (ms).
Save a single artifacts bundle (.pt) with weights and all preprocessing metadata so inference matches training exactly.
2) Project Layout
Single-file implementation (for speed during a prototype):
your_script.py
recordings/              # put FSDD wavs here (e.g., 0_jackson_0.wav)
fsdd_cnn_mel.pt          # saved after training (weights + preprocessing bundle)
3) Build / Setup
Python 3.9+ recommended.
# PyTorch (CPU-only example). If you have CUDA, install the matching wheel instead.
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Audio + ML stack
pip install librosa==0.10.1 numpy scikit-learn
macOS tip (if Librosa complains about I/O): brew install libsndfile
Device selection
The script chooses CUDA if available. On Apple Silicon, you can switch to MPS:
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
4) How to Run
A) Train
python your_script.py train --data_dir recordings
Outputs:
Epoch logs with train/val loss & accuracy
Artifacts saved to fsdd_cnn_mel.pt (configurable at top of file)
Key knobs (top of file):
Training: EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, SEED
Audio: N_MELS, N_FFT, HOP_LEN, WIN_LEN, FMIN, FMAX
Augment: NOISE_PROB, NOISE_LEVEL
B) Inference on a Single File
python your_script.py infer path/to/clip.wav --artifacts fsdd_cnn_mel.pt
Outputs:
Predicted digit and class probabilities array
C) Robustness Suite (Noise + Time-Shift)
python your_script.py robust --data_dir recordings --artifacts fsdd_cnn_mel.pt \
  --noise_levels 0.002 0.005 0.01 0.02 \
  --shifts_ms 0 10 20 40
What it does:
Rebuilds the same test split using the saved seed
Prints clean test accuracy
Prints accuracy across specified white-noise std levels and time shifts
5) Testing the App
Quick functional test
Run train and confirm it prints epochs and saves fsdd_cnn_mel.pt.
Pick any test file and run infer to ensure preprocessing + forward pass works.
Run robust and confirm you get a clean baseline accuracy and the sweeps.
Sanity checks you’ll see in logs
“Found N files. Example: …”
“Model params: < 100,000”
“Test acc (clean): …”
Confusion matrix + classification report
6) Results (Example)
These are representative numbers for clean FSDD with this setup. Your exact results will vary by split and hardware, but the baseline is designed to be strong and stable.
Setting	Accuracy
Clean test (no aug)	~0.95–0.99
White noise σ=0.005	~0.92–0.98
White noise σ=0.010	~0.88–0.96
Time shift 10 ms	≈ clean (small drop)
Time shift 40 ms	mild drop (still robust)
Per-class behavior
Confusion tends to happen between acoustically similar digits for certain speakers (“nine” vs “five” depending on pronunciation). The classification report will surface any weak classes; nudging NOISE_LEVEL/NOISE_PROB or adding a small time-shift augmentation during training typically evens this out.
If your validation accuracy lags: lower NOISE_LEVEL, increase EPOCHS slightly (e.g., 30), or add a third DS-Conv block with a tiny channel bump (e.g., 64→96) while keeping params modest.
7) Reproducibility & Artifacts
The saved bundle fsdd_cnn_mel.pt contains:
state_dict (model weights)
mel_cfg: {n_mels, n_fft, hop_length, win_length, fmin, fmax}
target_T: temporal length after pad/trim
channel_mean, channel_std: per-mel stats from train only
rms_target_dbfs
label_map
split_seed
Inference and robustness automatically use this bundle so preprocessing exactly matches training.
8) Troubleshooting
No WAV files found
Check --data_dir and file extensions. The script searches recursively for .wav/.WAV.
Unexpected filename
Must match digit_speaker_index.wav. Fix names or update NAME_RE in the script.
Librosa I/O error
Install libsndfile (macOS: brew install libsndfile).
Device runtime error
Use a PyTorch build that matches your hardware; switch DEVICE if needed.
Broadcasting error in standardization
Keep feature shapes as (N, C, T); mean/std are (C,) and are applied with mean[None, :, None].
9) Extensions (Optional)
Add time-shift augmentation during training (already implemented as a transform utility).
Swap to SpecAugment-lite (time/frequency masking) for a small robustness bump.
Replace the head with ArcFace or add label smoothing if you see over-confidence.
Export to ONNX for lightweight deployment.
10) License & Credits
Dataset: Free Spoken Digit Dataset (FSDD) — follow its license.
Libraries: PyTorch, Librosa, scikit-learn, NumPy.
