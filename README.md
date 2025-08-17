# Audio-Digit
FSDD Lightweight CNN — Audio Digit Classifier
A tiny, fast baseline that turns audio of spoken digits (0–9) into predictions using log-mel features and a depthwise-separable 1D-CNN. It trains in minutes, stays under ~100k params, and includes robustness checks (white noise + time-shift).
What this really means is: audio in → digit out, with clean structure, repeatable splits, and one command per task.
Features
Log-mel frontend (40 mels, 8 kHz native FSDD)
Depthwise-separable 1D-CNN + global average pooling
Train/val/test split with saved seed for reproducibility
Per-channel standardization computed on train only
On-the-fly white-noise augmentation for train subset
Robustness sweeps: Gaussian noise std and time shifts
Single-file inference that mirrors train preprocessing
Artifacts bundle (.pt) with weights + feature config + stats
Dataset
Use the Free Spoken Digit Dataset (FSDD) (8 kHz WAV). Put files under recordings/ and keep the naming pattern:
<digit>_<speaker>_<index>.wav
e.g., 0_jackson_0.wav, 9_yweweler_12.wav
The code validates filenames and will warn/skip unknown patterns.
Environment
Python >= 3.9
Recommended packages:
librosa
numpy
torch
scikit-learn
Quick install:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install librosa==0.10.1 numpy scikit-learn
If you have CUDA available, install the CUDA build of PyTorch instead of the CPU wheel above.
How it works (short version)
Load each WAV at native sample rate (FSDD is 8 kHz).
RMS-normalize to a target loudness (−20 dBFS).
(Train only) With a set probability, add small white noise.
Compute log-mel spectrograms, center pad/truncate to a shared T.
Compute per-mel mean/std on the train set and standardize all splits.
Train a tiny DS-Conv 1D-CNN → GAP → Linear(10).
Early-stop on validation accuracy. Save the best state + preprocessing config.
Commands
All commands are run via the same script.
1) Train
python your_script.py train --data_dir recordings
Output:
Prints epoch logs and val accuracy
Saves fsdd_cnn_mel.pt with:
state_dict, mel_cfg, target_T
channel_mean, channel_std
rms_target_dbfs, label_map, split_seed
Training knobs (top of file):
EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY
NOISE_PROB, NOISE_LEVEL

3) Single-file inference
python Audio-digit.py infer path/to/clip.wav --artifacts fsdd_cnn_mel.pt
Output:
Predicted digit + class probabilities

4) Robustness evaluation
python your_script.py robust --data_dir recordings --artifacts fsdd_cnn_mel.pt \
  --noise_levels 0.002 0.005 0.01 0.02 \
  --shifts_ms 0 10 20 40
   
What it does:
Rebuilds the same test split using the saved seed
Evaluates clean test accuracy
Sweeps white noise std and time-shift (ms) and reports accuracy

Expected results
On clean FSDD, a setup like this typically lands >95% test accuracy. Noise and large shifts will gracefully degrade performance; the robustness sweep quantifies that drop.
Project layout (single-file script)
Everything lives in one file for speed:
Config + constants
Audio utils (RMS normalize, noise add, time shift)
Feature extraction (log-mel) + pad/trim
Dataset scan + featurization
Train/val/test split + stats
Tiny DS-Conv model
Train/eval loops, confusion matrix + report
CLI for train / infer / robust
If you later want to split modules, the boundaries are already clean: audio_utils, features, data, model, train, eval.


Notes & tips
Device selection: the file auto-uses CUDA if available (DEVICE = "cuda" if torch.cuda.is_available() else "cpu").
On Apple Silicon, consider changing to:
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
Standardization shapes: features have shape (N, C, T) with C = N_MELS. Mean/std are (C,) and are broadcast as X_norm = (X - mean[None, :, None]) / std[None, :, None]. If you edit code, keep these dimensions consistent to avoid broadcasting errors.
Target length (target_T): derived from the first file’s mel length. All examples are center-padded/truncated to match.
Augmentation policy: noise is applied after RMS normalization, train only. Tune NOISE_PROB/NOISE_LEVEL if val performance drops.
Filename parsing: strict pattern protects your labels. If you have custom names, update NAME_RE.
Troubleshooting
“No WAV files found…”
Check --data_dir path and that it contains .wav files (case-insensitive). Ensure files aren’t nested deeper than expected.
“Unexpected filename” warnings
Filenames must look like 3_speaker_12.wav. Fix names or update NAME_RE.
Librosa errors about sndfile
Install libsndfile (e.g., brew install libsndfile on macOS).
Device errors (CUDA/MPS)
Use the right PyTorch build for your hardware, and switch DEVICE accordingly.
Metrics & artifacts
After training completes, you’ll see:
Clean test accuracy
Confusion matrix (rows=true, cols=pred)
Classification report (per-class precision/recall/F1)
Saved artifacts at fsdd_cnn_mel.pt (path configurable)
Acknowledgments
Dataset: Free Spoken Digit Dataset (FSDD)
Libraries: PyTorch, Librosa, scikit-learn, NumPy
