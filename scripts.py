#!/usr/bin/env python3
import json
from pathlib import Path
import soundfile as sf

# 1. 讀混合設定檔
cfg = json.load(open('filelists/0511_vad_test/mixed.json','r', encoding='utf-8'))

# 2. 拿出 training_files（可能在 data 底下，也可能在 root）
tf = cfg.get('data', {}).get('training_files', cfg.get('training_files'))
if isinstance(tf, str):
    training_files = [tf]
else:
    training_files = tf

# 3. 走每個 .txt 列表，讀每行 wav 路徑
wav_lengths = []
for txtlist in training_files:
    txt_path = Path(txtlist)
    if not txt_path.exists():
        print(f"Warning: list file not found: {txt_path}")
        continue
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            wavpath = line.split('|',1)[0]
            wavfile = Path(wavpath)
            if not wavfile.exists():
                print(f"  skip missing wav: {wavfile}")
                continue
            data, sr = sf.read(str(wavfile))
            wav_lengths.append(len(data))

# 4. 統計並印出
if wav_lengths:
    wav_lengths.sort()
    N = len(wav_lengths)
    def pct(p): return wav_lengths[int(N*p)]
    print(f"Total files: {N}")
    print(f"Min length: {wav_lengths[0]} samples")
    print(f"25% percentile: {pct(0.25)}")
    print(f"Median: {pct(0.5)}")
    print(f"75% percentile: {pct(0.75)}")
    print(f"Max length: {wav_lengths[-1]} samples")
else:
    print("No wav files found.")
