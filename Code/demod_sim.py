# demod_sim.py
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
import sys
import matplotlib.pyplot as plt

# usage: python demod_sim.py attack_am_23k.wav demod_sim.wav
infile = '/content/attack_am_strong_23k.wav'
outfile = '/content/demod_sim.wav'

attack, sr = sf.read(infile)
# simulate nonlinearity: square (2nd order) and small cubic term
nonlin = attack**2 + 0.001 * attack**3

# low-pass filter to recover audible band (<9 kHz)
def lowpass(x, sr, cutoff=9000):
    b, a = butter(6, cutoff/(sr/2), btype='low')
    return filtfilt(b, a, x)

demod = lowpass(nonlin, sr, cutoff=9000)
demod = demod / (np.max(np.abs(demod)) + 1e-9) * 0.9

sf.write(outfile, demod, sr)
print("Wrote demodulated:", outfile)

# Optional: save plots for report
t = np.arange(len(attack))/sr
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t[:5000], attack[:5000])
plt.title("Attack (time)")
plt.subplot(3,1,2)
plt.plot(t[:5000], nonlin[:5000])
plt.title("After nonlinearity (time)")
plt.subplot(3,1,3)
plt.plot(t[:5000], demod[:5000])
plt.title("Demodulated (time)")
plt.tight_layout()
plt.savefig("phase1_waveforms.png")
print("Saved phase1_waveforms.png")
