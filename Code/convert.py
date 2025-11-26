#convert.py
This file contains two conversion blocks:
import librosa
import soundfile as sf
import sys

# usage: python convert.py hey_google.mp3 hey_google_96k.wav
infile = '/content/Hey google1.m4a'
outfile = '/content/hey_google_96k.wav'

y, sr = librosa.load(infile, sr=None, mono=True)   # load original sr
y96 = librosa.resample(y, orig_sr=sr, target_sr=96000)
sf.write(outfile, y96, 96000)
print("Saved", outfile, "sr=96000")

# second conversion
import librosa
import soundfile as sf
import sys

infile = '/content/Noise.m4a'
outfile = '/content/noise.wav'

y, sr = librosa.load(infile, sr=None, mono=True)
y96 = librosa.resample(y, orig_sr=sr, target_sr=96000)
sf.write(outfile, y96, 96000)
print("Saved", outfile, "sr=96000")
