# detector_adaptive.py
import numpy as np, soundfile as sf, sys
from scipy.signal import stft
import matplotlib.pyplot as plt

# usage: python detector_adaptive.py attack_am_strong_23k.wav
fname = 'attack_am_strong_23k.wav' if len(sys.argv)>1 else 'attack_am_strong_23k.wav'
x, sr = sf.read(fname)
print("File:", fname, "sr:", sr, "len(s):", len(x)/sr)

# compute STFT
f, t, Z = stft(x, fs=sr, nperseg=4096, noverlap=2048)
Zmag = np.abs(Z)
# focus band > 18kHz and <= fc+5k to avoid weird aliasing
band_mask = (f >= 18000) & (f <= min(34000, sr/2-1))
band_energy_frames = np.mean(Zmag[band_mask,:]**2, axis=0)

# statistics
median_e = np.median(band_energy_frames)
mean_e = np.mean(band_energy_frames)
std_e = np.std(band_energy_frames)
p90 = np.percentile(band_energy_frames, 90)
p95 = np.percentile(band_energy_frames, 95)

print("Band-energy frames: mean {:.6e}, median {:.6e}, std {:.6e}, p90 {:.6e}, p95 {:.6e}".format(mean_e, median_e, std_e, p90, p95))

# adaptive threshold options (choose one)
thr1 = median_e + 4 * std_e         # median + k*std
thr2 = p95 * 0.75                   # 75% of 95th percentile (tunable)
thr3 = median_e * 8                 # multiplier of median (tunable)

# choose threshold (pick thr1 usually robust)
threshold = thr2
print("Using threshold =", threshold)

# detection
peaks = band_energy_frames > threshold
num_detected = int(np.sum(peaks))
print("Detected frames:", num_detected, "/", len(peaks))

# print times of detected frames
det_times = t[peaks]
print("Detected at times (s):", det_times.tolist())

# plot
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.pcolormesh(t, f, 20*np.log10(Zmag+1e-12), shading='gouraud')
plt.ylim(0, 30000)
plt.ylabel('Freq (Hz)')
plt.title('Spectrogram (dB)')

plt.subplot(2,1,2)
plt.plot(t, band_energy_frames, label='band energy (>18kHz)')
plt.axhline(threshold, color='r', linestyle='--', label='threshold')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend()
plt.tight_layout()
plt.savefig('detector_adaptive_output.png')
print("Saved detector_adaptive_output.png")
