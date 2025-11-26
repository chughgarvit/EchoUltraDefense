# detect_and_block.py
# Usage: python detect_and_block.py input.wav output_blocked.wav
import sys
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

infile = "attack_am_strong_23k.wav"
outfile = "blocked.wav"

x, sr = sf.read(infile)
if x.ndim>1:
    x = x.mean(axis=1)

# STFT params
nperseg = 4096
noverlap = 2048
f, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
Zmag = np.abs(Z)

# band-energy in ultrasonic band
band_mask = (f >= 18000) & (f <= 34000)
band_energy = np.mean(Zmag[band_mask,:]**2, axis=0)

# adaptive threshold (percentile-based)
threshold = np.percentile(band_energy, 95) * 0.75
peaks = band_energy > threshold
print(f"Threshold={threshold:.6e}, detected_frames={np.sum(peaks)}/{len(peaks)}")

# Create blocked copy
x_blocked = x.copy()
hop = nperseg - noverlap
T_block_ms = 350
T_block_samples = int(T_block_ms * sr / 1000)

# For each detected stft frame, mute the corresponding sample region
for idx, detected in enumerate(peaks):
    if detected:
        start_sample = idx * hop
        end_sample = min(len(x_blocked), start_sample + T_block_samples)
        x_blocked[start_sample:end_sample] = 0.0

sf.write(outfile, x_blocked, sr)
print(f"Saved blocked file: {outfile}")

# optional save simple waveform plot
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    t_s = np.arange(len(x))/sr
    plt.plot(t_s, x, label='original', alpha=0.6)
    plt.plot(t_s, x_blocked, label='blocked', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("blocked_waveform.png")
    print("Saved blocked_waveform.png")
except Exception as e:
    print("Could not save waveform:", e)
