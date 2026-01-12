#!/usr/bin/env python3
"""
批次重採樣音頻檔案到目標採樣率
用於修復 ESD 語料庫中 16kHz 與 22050Hz 混合的問題
"""

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TARGET_SR = 22050  # 目標採樣率


def get_sample_rate(filepath):
    """獲取音頻檔案的採樣率"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=sample_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', filepath],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return None


def resample_file(filepath, target_sr=TARGET_SR):
    """重採樣單個音頻檔案（原地覆蓋）"""
    current_sr = get_sample_rate(filepath)
    
    if current_sr is None:
        return filepath, "error", "無法讀取採樣率"
    
    if current_sr == target_sr:
        return filepath, "skip", f"已經是 {target_sr}Hz"
    
    # 創建臨時檔案
    temp_path = filepath + ".temp.wav"
    
    try:
        # 使用 ffmpeg 重採樣
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', filepath, '-ar', str(target_sr), temp_path],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return filepath, "error", result.stderr
        
        # 替換原檔案
        os.replace(temp_path, filepath)
        
        # 刪除對應的 .spec.pt 快取檔案（如果存在）
        spec_cache = filepath.replace(".wav", ".spec.pt")
        if os.path.exists(spec_cache):
            os.remove(spec_cache)
        
        return filepath, "success", f"{current_sr}Hz → {target_sr}Hz"
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return filepath, "error", str(e)


def main(filelist_path):
    """主函數：從 filelist 讀取音頻路徑並批次處理"""
    
    # 讀取 filelist
    audio_files = []
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if parts:
                audio_files.append(parts[0])
    
    print(f"📂 共找到 {len(audio_files)} 個音頻檔案")
    
    # 先掃描需要重採樣的檔案
    print("🔍 掃描採樣率...")
    need_resample = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_sample_rate, f): f for f in audio_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="掃描中"):
            filepath = futures[future]
            sr = future.result()
            if sr is not None and sr != TARGET_SR:
                need_resample.append(filepath)
    
    print(f"\n⚠️  需要重採樣的檔案數: {len(need_resample)}")
    
    if not need_resample:
        print("✅ 所有檔案已經是目標採樣率，無需轉換！")
        return
    
    # 確認是否繼續
    response = input(f"\n是否開始重採樣 {len(need_resample)} 個檔案? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 批次重採樣
    print("\n🔄 開始重採樣...")
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(resample_file, f): f for f in need_resample}
        for future in tqdm(as_completed(futures), total=len(futures), desc="轉換中"):
            filepath, status, msg = future.result()
            if status == "success":
                success_count += 1
            elif status == "error":
                error_count += 1
                print(f"\n❌ 錯誤: {filepath} - {msg}")
    
    print(f"\n✅ 完成！成功: {success_count}, 失敗: {error_count}")
    print("⚠️  請刪除所有 .spec.pt 快取檔案後重新訓練")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python resample_audio.py <filelist_path>")
        print("範例: python resample_audio.py dataset/emotion_train.txt")
        sys.exit(1)
    
    main(sys.argv[1])
