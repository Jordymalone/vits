#!/usr/bin/env python3
"""
自動生成包含情緒 ID 的 filelist
整合 main_continue.py 的標點清洗與 beji_cut 音素轉換邏輯
"""

import os
import sys
import glob
import argparse
import re
import random

# --- 強制加入專案根目錄到路徑，解決 ImportError ---
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

try:
    from tools.phonemes_transformation.zh.beji_cut_vowel_corpus import zh_frontend
except ImportError as e:
    print(f"❌ 錯誤：無法載入音素轉換模組。詳情: {e}")
    print(f"請確保您是在 vits 專案根目錄執行：python3 tools/prepare_emotion_filelist.py")
    sys.exit(1)

# 情緒映射
EMOTION_MAP = {
    'Neutral': 0, 'Happy': 1, 'Sad': 2, 'Angry': 3,
    'Surprise': 4, 'Fear': 5, 'Disgust': 6
}

class TextProcessor:
    """完全移植自 main_continue.py 的文本處理邏輯"""
    def __init__(self):
        self.beji_cut = zh_frontend()

    def to_half_width(self, text):
        converted_text = ''
        for char in text:
            ascii_code = ord(char)
            if 65281 <= ascii_code <= 65374:
                converted_char = chr(ascii_code - 65248)
            else:
                converted_char = char
            converted_text += converted_char
        return converted_text

    def contains_only_chinese_and_numbers(self, text):
        """檢查是否只含中文與數字 (來自 main_continue.py)"""
        alpha = re.findall(r'[a-zA-Z]', text)
        return not bool(alpha)

    def content_punctuation(self, text):
        """標點符號清洗與標準化 (來自 main_continue.py)"""
        texted_process = self.to_half_width(text)
        texted_process = re.sub(r'(\W)\1+', r'\1 ', texted_process)
        texted_process = texted_process.replace('\\n', '。')

        # 移除 ESD 常見的特殊符號
        texted_process = texted_process.replace('…', '').replace('...', '')
        # 移除或替換頓號 → 頓號會被 beji_cut 誤轉成 "、1" 和 "1、" 音素
        texted_process = texted_process.replace('、', '，')  # 替換為逗號

        # 移除各種破折號 (會被 beji_cut 誤轉成 "―1" "1―" "—1" "1—" 等音素)
        # U+2014 (EM DASH): —, U+2015 (HORIZONTAL BAR): ―
        for dash in ['――', '——', '―', '—', '--']:
            texted_process = texted_process.replace(dash, '，')

        for char in '」」「「『』《》()[]（）':
            texted_process = texted_process.replace(char, "")

        texted_process = texted_process.strip()
        # 確保有結尾標點
        if texted_process and not texted_process.endswith(('。', '!', '?', '！', '？')):
            texted_process += '。'

        return texted_process

    def get_phonemes(self, text):
        # 1. 過濾包含英文的句子
        if not self.contains_only_chinese_and_numbers(text):
            return None
            
        # 2. 標點符號處理
        cleaned_text = self.content_punctuation(text)
        if len(cleaned_text) < 3: # 參考 main_continue 邏輯
            return None
            
        # 3. 音素轉換 (beji_cut)
        result, status = self.beji_cut.get_phonemes(cleaned_text)
        return result if status else None

def generate_filelist(data_root, output_file, language='ZH', train_ratio=0.9):
    tp = TextProcessor()
    train_lines = []
    val_lines = []
    file_count = 0

    print(f"Scanning dataset at: {data_root}")

    speakers = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    for speaker in speakers:
        speaker_path = os.path.join(data_root, speaker)
        ids = re.findall(r'\d+', speaker)
        speaker_id = int(ids[-1]) if ids else hash(speaker) % 1000

        emotion_folders = sorted([d for d in os.listdir(speaker_path) if os.path.isdir(os.path.join(speaker_path, d))])

        for folder in emotion_folders:
            eid = next((i for name, i in EMOTION_MAP.items() if name.upper() in folder.upper()), None)
            if eid is None: continue

            wav_files = sorted(glob.glob(os.path.join(speaker_path, folder, "*.wav")))
            # 加入隨機打散，確保 train/val 分布均勻
            random.shuffle(wav_files)

            for i, wav_path in enumerate(wav_files):
                txt_path = wav_path.replace('.wav', '.txt')
                if not os.path.exists(txt_path): continue

                with open(txt_path, 'r', encoding='utf-8') as tf:
                    raw_text = tf.read().strip()

                # 執行與 main_continue 一致的處理
                phonemes = tp.get_phonemes(raw_text)
                
                if phonemes:
                    line = f"{wav_path}|{speaker_id}|{language}|{phonemes}|{eid}\n"
                    if i < len(wav_files) * train_ratio:
                        train_lines.append(line)
                    else:
                        val_lines.append(line)
                    file_count += 1

    # 寫入結果
    with open(f"{output_file}_train.txt", 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(f"{output_file}_val.txt", 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f"\n✅ 成功產生 Filelist！總筆數: {file_count}")
    print(f"📊 數據分布:")
    for name, eid in sorted(EMOTION_MAP.items(), key=lambda x: x[1]):
        count = sum(1 for l in train_lines+val_lines if l.rstrip().endswith(f"|{eid}"))
        if count > 0: print(f"   {name:10}: {count} 筆")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output', type=str, default='dataset/emotion')
    parser.add_argument('--language', type=str, default='ZH')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_filelist(args.data_root, args.output, args.language, args.train_ratio)