# analyze_band.py
import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from scipy.signal import stft, welch
import sys

fname = 'attack_am_strong_23k.wav' if len(sys.argv)>1 else 'attack_am_strong_23k.wav'
x, sr = sf.read(fname)
print("File:", fname, "sr:", sr, "len(s):", len(x)/sr)

# power spectral density (Welch) for global view
f, Pxx = welch(x, fs=sr, nperseg=8192)
# compute mean energy above 18 kHz
mask = f >= 18000
band_energy = np.mean(Pxx[mask])
total_energy = np.mean(Pxx)
print("PSD band energy (>18kHz):", band_energy)
print("PSD total mean energy:", total_energy)
print("Band/Total ratio:", band_energy/ (total_energy+1e-12))

# short-time spectrogram (STFT)
f2, t2, Z = stft(x, fs=sr, nperseg=4096, noverlap=2048)
Zmag = np.abs(Z)
band_mask = f2 >= 18000
band_energy_frames = np.mean(Zmag[band_mask,:]**2, axis=0)

print("Mean band energy (STFT):", band_energy_frames.mean())
print("Max band energy (STFT):", band_energy_frames.max())

# plot spectrogram (log) and band energy
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.pcolormesh(t2, f2, 20*np.log10(Zmag+1e-12), shading='gouraud')
plt.ylim(0, 30000)
plt.colorbar(label='Magnitude (dB)')
plt.title('Spectrogram (up to 30kHz)')
plt.ylabel('Freq (Hz)')

plt.subplot(2,1,2)
plt.plot(t2, band_energy_frames)
plt.axhline(band_energy_frames.mean()*4, color='r', linestyle='--', label='threshold x4')
plt.xlabel('Time (s)')
plt.ylabel('Band energy (>18kHz)')
plt.legend()
plt.tight_layout()
plt.savefig('attack_spectrogram_and_band.png')
print("Saved attack_spectrogram_and_band.png")
