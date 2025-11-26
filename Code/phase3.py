# phase3.py
import os, numpy as np, soundfile as sf, librosa, pandas as pd
from scipy.signal import stft, fftconvolve, butter, filtfilt
import matplotlib.pyplot as plt

# ---------------------- File Inputs ---------------------
attack_file = "attack_am_strong_23k.wav"
ref_file    = "hey_google_96k.wav"
normal_files = ["Normal1cam.m4a", "Normalflash.m4a", "Normalset.m4a"]

# convert m4a → wav 96k
def ensure_wav_96k(fname):
    y, sr = librosa.load(fname, sr=96000)
    out = fname.replace(".m4a", "_96k.wav")
    sf.write(out, y, 96000)
    return out

normal_files = [ensure_wav_96k(f) if f.endswith(".m4a") else f for f in normal_files]

# --------------------- Utility Functions ---------------------
def load_mono(fname, sr_target=96000):
    y, sr = librosa.load(fname, sr=sr_target)
    if y.ndim>1: y=y.mean(axis=1)
    return y, sr_target

def demodulate_signal(x, sr):
    y = x**2
    y_low = librosa.resample(y, orig_sr=sr, target_sr=8000)
    y_back = librosa.resample(y_low, orig_sr=8000, target_sr=sr)
    return y_back[:len(y)]

def norm_corr(a,b):
    a=a-np.mean(a); b=b-np.mean(b)
    if np.std(a)==0 or np.std(b)==0: return 0.0
    corr=fftconvolve(a,b[::-1],mode='valid')
    return np.max(np.abs(corr))/(np.sqrt(np.sum(a*a)*np.sum(b*b))+1e-12)

def run_asr_proxy(infile, refname):
    x,sr = load_mono(infile)
    ref,_ = load_mono(refname)
    x_dem, ref_dem = demodulate_signal(x,sr), demodulate_signal(ref,sr)
    L = len(ref_dem); best=0; step=int(sr*0.01)
    for st in range(0,max(1,len(x_dem)-L+1),step):
        best=max(best,norm_corr(x_dem[st:st+L],ref_dem))
    return best,(best>=0.08)

def defense_detect_and_block(infile,outfile,lp_cut=10000):
    x,sr = load_mono(infile)
    f,t,Z = stft(x,fs=sr,nperseg=4096,noverlap=2048)
    Zmag=np.abs(Z)
    band = np.mean(Zmag[(f>=18000)&(f<=34000),:]**2, axis=0)
    thr = max(np.percentile(band,95)*0.75,1e-10)
    peaks=band>thr
    hop=4096-2048; block=int(0.35*sr)
    regions=[]; i=0
    while i<len(peaks):
        if peaks[i]:
            start=int(t[i]*sr); j=i
            while j+1<len(peaks) and peaks[j+1]: j+=1
            end=int(min(len(x),int(t[j]*sr)+block))
            regions.append([start,end]); i=j+1
        else: i+=1
    xb=x.copy()
    for s,e in regions: xb[s:e]=0
    b,a=butter(6,lp_cut/(sr/2),'low'); xb=filtfilt(b,a,xb)
    sf.write(outfile,xb,sr)
    return regions

def add_noise(x,n,snr):
    if snr==999: return x
    n = librosa.util.fix_length(data=n, size=len(x))   # ✅ fixed librosa API
    return x + n*np.sqrt(np.mean(x*x))/(10**(snr/20)/np.sqrt(np.mean(n*n)))

def stretch_speed(x,rate):
    return librosa.resample(x,orig_sr=96000,target_sr=int(96000*rate))

# ------------------- Simulation Settings ---------------------
SNR_list=[999,20,15,10,5]
atten_dB=[0,-3,-6,-12]
speeds=[1.00,0.95,1.05]
N_trials=2

x_base,_ = load_mono(attack_file)
n_base=np.random.randn(len(x_base))

results=[]

# ------------------- Full Loop --------------------------
for snr in SNR_list:
  for att in atten_dB:
    for sp in speeds:
      for tr in range(N_trials):
        x,_ = load_mono(attack_file)
        x = x*(10**(att/20))
        x = stretch_speed(x,sp)
        x = add_noise(x,n_base,snr)
        sf.write("temp.wav",x,96000)

        s1,det1 = run_asr_proxy("temp.wav",ref_file)
        regions = defense_detect_and_block("temp.wav","blocked.wav")
        s2,det2 = run_asr_proxy("blocked.wav",ref_file)

        latency = (regions[0][0]/96000*1000) if regions else 999
        results.append([snr,att,sp,tr,det1,det2,latency])

df=pd.DataFrame(results,columns=["SNR","Atten","Speed","Trial","NoDefense","Defense","Latency(ms)"])
df.to_csv("Phase3_results.csv",index=False)
print(df)

# ----------------Plot-----------------
summary=df.groupby("SNR")[["NoDefense","Defense"]].mean()*100
plt.figure(figsize=(5,4))
plt.plot(summary,"o-")
plt.title("Phase-3: Attack Success vs Noise")
plt.xlabel("SNR (dB)"); plt.ylabel("Attack Success (%)")
plt.grid(); plt.legend(["No Defense","With Defense"])
plt.tight_layout(); plt.savefig("P3_attack_plot.png")

print("\n✅ Phase-3 Complete — results saved (CSV + plot)")

