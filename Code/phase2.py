# phase2.py
import os, numpy as np, soundfile as sf, librosa, pandas as pd
from scipy.signal import stft, fftconvolve
import matplotlib.pyplot as plt



def run_asr_proxy(infile, refname):
    x,sr = load_mono(infile)
    ref,sr_ref = load_mono(refname)
    x_dem = demodulate_signal(x,sr)
    ref_dem = demodulate_signal(ref,sr)
    Lr = len(ref_dem)
    best = 0.0
    step = int(sr*0.01)
    for start in range(0, max(1,len(x_dem)-Lr+1), step):
        seg = x_dem[start:start+Lr]
        score = norm_corr(seg, ref_dem)
        best = max(best,score)
    THR = 0.08
    return best, (best>=THR)


def defense_detect_and_block(infile, outfile, sr_target=96000,
                             nperseg=4096, noverlap=2048,
                             block_ms=400, lowpass_after=True, lp_cut=10000):
    # read
    x, sr = sf.read(infile)
    if x.ndim > 1:
        x = x.mean(axis=1)
    # compute STFT band energy
    f, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Zmag = np.abs(Z)
    band_mask = (f >= 18000) & (f <= 34000)
    band_energy = np.mean(Zmag[band_mask,:]**2, axis=0)
    #thr = np.percentile(band_energy, 95) * 0.75
    thr = max(np.percentile(band_energy, 95) * 0.75, 1e-10)
    peaks = band_energy > thr

    # merge consecutive detections into regions (avoid small gaps)
    hop = nperseg - noverlap
    block_samps = int(block_ms * sr / 1000)
    merged_regions = []
    i = 0
    while i < len(peaks):
        if peaks[i]:
            # start region from this frame's start
            start = int(t[i]*sr)
            # extend while consecutive frames are detected
            j = i
            while j+1 < len(peaks) and peaks[j+1]:
                j += 1
            # set region end to end of j plus block_samps margin
            end = int(min(len(x), int(t[j]*sr) + block_samps))
            # merge with previous if overlapping or close (within hop)
            if merged_regions and start <= merged_regions[-1][1] + hop:
                merged_regions[-1][1] = max(merged_regions[-1][1], end)
            else:
                merged_regions.append([start, end])
            i = j + 1
        else:
            i += 1

    # apply strong mute to merged regions
    x_blocked = x.copy()
    for (s,e) in merged_regions:
        x_blocked[s:e] = 0.0

    # optional low-pass to further remove residual HF energy
    if lowpass_after:
        b,a = butter(6, lp_cut/(sr/2), btype='low')
        x_blocked = filtfilt(b,a,x_blocked)

    sf.write(outfile, x_blocked, sr)
    return merged_regions, thr

# --------------------------------------------------
# Step B: run batch simulation
# --------------------------------------------------
attack_file="attack_am_strong_23k.wav"
ref_file="hey_google_96k.wav"
noise_file="/content/noise.wav"

x_attack,sr=load_mono(attack_file)
n_base=np.random.randn(len(x_attack)) if noise_file is None else load_mono(noise_file)[0]

SNR_list=[999,20,15,10,5]
atten_dB_list=[0,-3,-6,-12]
N_trials=3

def add_noise(x,n,snr_db):
    if snr_db==999: return x
    n=librosa.util.fix_length(data=n,size=len(x))
    rms_x=np.sqrt(np.mean(x**2))
    rms_n=np.sqrt(np.mean(n**2))
    target_rms_n=rms_x/(10**(snr_db/20))
    n_scaled=n*(target_rms_n/(rms_n+1e-12))
    return x+n_scaled

results=[]
for snr in SNR_list:
    for atten_db in atten_dB_list:
        for trial in range(N_trials):
            x_var=x_attack*(10**(atten_db/20))
            x_var=add_noise(x_var,n_base,snr)
            tmp_in=f"in_{snr}_{atten_db}_{trial}.wav"
            tmp_out=f"out_{snr}_{atten_db}_{trial}.wav"
            sf.write(tmp_in,x_var,sr)
            # without defense
            score1,det1=run_asr_proxy(tmp_in,ref_file)
            # with defense
            defense_detect_and_block(tmp_in,tmp_out)
            score2,det2=run_asr_proxy(tmp_out,ref_file)
            results.append([snr,atten_db,trial,det1,det2,score1,score2])
            print(f"SNR={snr},Atten={atten_db},trial={trial} -> {det1}->{det2}")

df=pd.DataFrame(results,columns=["snr","atten_db","trial","success_no_defense","success_with_defense","score_no_defense","score_with_defense"])
df.to_csv("results.csv",index=False)
print("\n✅ Saved results.csv")

