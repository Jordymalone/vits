#!/usr/bin/env python3
"""
從 emotion_train.txt 和 emotion_val.txt 提取所有音素
生成 emotion_phones.txt 供 symbols.py 使用
"""
import os
import sys

def extract_phones(train_file, val_file, output_file):
    """提取訓練集與驗證集中的所有唯一音素"""
    total_ph = []

    print(f"📖 讀取訓練集: {train_file}")
    with open(train_file, 'r', encoding='utf8') as f:
        in_train = f.readlines()

    for line in in_train:
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        # 格式: wav_path|speaker_id|language|phonemes|emotion_id
        # 音素在第 3 個位置 (index=3)
        ph = parts[3].strip()
        syllables = ph.split(" ")
        for sy in syllables:
            sy = sy.strip()
            if sy and sy not in total_ph:
                total_ph.append(sy)
                print(f"  發現新音素: {sy}")

    print(f"\n📖 讀取驗證集: {val_file}")
    with open(val_file, 'r', encoding='utf8') as f:
        in_val = f.readlines()

    for line in in_val:
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        ph = parts[3].strip()
        syllables = ph.split(" ")
        for sy in syllables:
            sy = sy.strip()
            if sy and sy not in total_ph:
                total_ph.append(sy)
                print(f"  發現新音素: {sy}")

    # 按照字母順序排序音素（保持訓練一致性）
    total_ph.sort()

    # 寫入音素字典
    with open(output_file, 'w', encoding='utf8') as f:
        for i, ph in enumerate(total_ph):
            f.write(f'{i} {ph}\n')

    print(f"\n✅ 成功生成音素字典: {output_file}")
    print(f"📊 總音素數量: {len(total_ph)}")
    return total_ph

if __name__ == "__main__":
    train_file = "../dataset/emotion_train.txt"
    val_file = "../dataset/emotion_val.txt"
    output_file = "../dataset/emotion_phones.txt"

    if not os.path.exists(train_file):
        print(f"❌ 錯誤: 找不到 {train_file}")
        sys.exit(1)
    if not os.path.exists(val_file):
        print(f"❌ 錯誤: 找不到 {val_file}")
        sys.exit(1)

    phones = extract_phones(train_file, val_file, output_file)

    print("\n💡 下一步:")
    print(f"   請將 dataset/emotion_phones.txt 中的音素更新到 text/symbols.py")
