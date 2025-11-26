# measure_similarity.py
import librosa, numpy as np, sys
from scipy.signal import correlate

orig, sr = librosa.load('/content/hey_google_96k.wav', sr=96000)
demod, _ = librosa.load('/content/demod_sim.wav', sr=96000)

# trim or align
L = min(len(orig), len(demod))
o = orig[:L]
d = demod[:L]

# normalize
o = o / (np.std(o)+1e-9)
d = d / (np.std(d)+1e-9)

corr = correlate(d, o, mode='valid')
score = np.max(np.abs(corr)) / L
print("Cross-corr peak (normalized):", score)
