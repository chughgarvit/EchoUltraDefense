# asr_check_nosmodel.py
# Usage: python asr_check_nosmodel.py input.wav reference_wake.wav
import sys, numpy as np, soundfile as sf
import librosa
from scipy.signal import fftconvolve

def load_mono(fname, sr_target=96000):
    y, sr = sf.read(fname)
    if y.ndim>1:
        y = y.mean(axis=1)
    if sr != sr_target:
        y = librosa.resample(y, sr, sr_target)
        sr = sr_target
    return y, sr

# simple demodulation proxy — square and lowpass via smoothing
# simple demodulation proxy — square and lowpass via smoothing
def demodulate_signal(x, sr):
    # simulate nonlinearity: square
    y = x**2
    # downsample then upsample to remove high frequencies (acts as a low-pass filter)
    y_low = librosa.resample(y, orig_sr=sr, target_sr=8000)
    y_back = librosa.resample(y_low, orig_sr=8000, target_sr=sr)
    # trim to same length
    if len(y_back) > len(y):
        y_back = y_back[:len(y)]
    return y_back

def norm_corr(a,b):
    # normalized cross-correlation peak between signals (zero mean)
    a = a - np.mean(a); b = b - np.mean(b)
    if np.std(a)==0 or np.std(b)==0:
        return 0.0
    corr = fftconvolve(a, b[::-1], mode='valid')
    denom = np.sqrt(np.sum(a*a)*np.sum(b*b))
    return np.max(np.abs(corr)) / (denom + 1e-12)

if __name__ == "__main__":
    infile = "/content/blocked.wav"
    refname = "hey_google_96k.wav"
    x, sr = load_mono(infile)
    ref, sr_ref = load_mono(refname)
    # demodulate both (simulated mic nonlinearity)
    x_dem = demodulate_signal(x, sr)
    ref_dem = demodulate_signal(ref, sr)
    # match lengths: search ref inside x_dem using sliding windows
    Lr = len(ref_dem)
    best = 0.0
    step = int(sr*0.01)  # slide every 10ms for speed
    for start in range(0, max(1,len(x_dem)-Lr+1), step):
        seg = x_dem[start:start+Lr]
        score = norm_corr(seg, ref_dem)
        if score > best:
            best = score
            best_start = start
    print(f"Best normalized correlation: {best:.4f}")

    # threshold (empirical) — choose 0.05..0.2 depending on experiments; set 0.08 default
    THR = 0.08
    if best >= THR:
        print("→ WAKE-WORD DETECTED (proxy). Score:", best)
    else:
        print("→ NO WAKE-WORD detected. Score:", best)
