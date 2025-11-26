# gen_attack_strong.py
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# ---------- PARAMETERS ----------
infile = 'hey_google_96k.wav'   # produced earlier
outfile = 'attack_am_strong_23k.wav'
sr = 96000

fc = 23000.0       # carrier frequency (Hz)
m = 1.0            # modulation index (0..1 typically). increase to 1.0 for strong modulation.
carrier_gain = 1.2 # extra gain on carrier ( >1 boosts ultrasonic energy)
pure_carrier_burst = True
burst_start_s = 0.05
burst_dur_s = 0.20
# --------------------------------

# helper lowpass to keep wake in voice band
from scipy.signal import butter, filtfilt
def lowpass(x, sr, cutoff=8000):
    b, a = butter(6, cutoff/(sr/2), btype='low')
    return filtfilt(b, a, x)

# load
wake, _ = librosa.load(infile, sr=sr, mono=True)
wake = wake / (np.max(np.abs(wake)) + 1e-12)
wake_lp = lowpass(wake, sr, cutoff=8000)

t = np.arange(len(wake)) / sr
carrier = np.cos(2 * np.pi * fc * t)

# construct AM signal
attack = (1.0 + m * wake_lp) * carrier

# boost ultrasonic energy by multiplying carrier component
attack = attack + (carrier_gain - 1.0) * carrier

# optionally add a short pure-carrier burst at start (helps detector see ultrasonic energy)
if pure_carrier_burst:
    start = int(burst_start_s * sr)
    end = min(len(attack), start + int(burst_dur_s * sr))
    attack[start:end] = attack[start:end] + 0.6 * carrier[start:end]  # add burst

# normalize (avoid clipping)
maxv = np.max(np.abs(attack)) + 1e-12
attack = attack / maxv * 0.95

sf.write(outfile, attack, sr)
print("Wrote:", outfile)
