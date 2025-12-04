import os

def format_duration(milliseconds):
    """將總毫秒數轉換為 時:分:秒.毫秒 的格式"""
    if not isinstance(milliseconds, (int, float)) or milliseconds < 0:
        return "00:00:00.000"

    # 計算總秒數
    total_seconds = milliseconds / 1000

    # 計算時、分、秒
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # 取得剩餘的毫秒
    remaining_milliseconds = int((seconds - int(seconds)) * 1000)

    # 格式化輸出字串
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{remaining_milliseconds:03d}"

def calculate_total_duration(file_path):
    """
    讀取指定的檔案，計算所有音檔的總時長。
    檔案格式應為：filepath|duration_in_ms|...
    """
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 '{file_path}'")
        return

    total_milliseconds = 0
    processed_lines = 0
    error_lines = 0

    print(f"正在讀取檔案：{file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 去除行尾的換行符並跳過空行
            line = line.strip()
            if not line:
                continue

            # 使用'|'分割字串
            parts = line.split('|')

            # 檢查格式是否正確 (至少要有兩個部分)
            if len(parts) >= 2:
                try:
                    # 第二部分是時長，將其轉換為整數並累加
                    duration_ms = int(parts[1])
                    total_milliseconds += duration_ms
                    processed_lines += 1
                except ValueError:
                    # 如果第二部分不是有效的數字
                    print(f"警告：第 {i+1} 行格式錯誤，無法解析時長：'{line}'")
                    error_lines += 1
            else:
                print(f"警告：第 {i+1} 行格式不符，已跳過：'{line}'")
                error_lines += 1

    print("\n--- 計算完成 ---")
    print(f"成功處理的行數：{processed_lines}")
    print(f"格式錯誤/跳過的行數：{error_lines}")
    print(f"總時長 (毫秒)：{total_milliseconds}")
    print(f"總時長 (時:分:秒.毫秒)：{format_duration(total_milliseconds)}")


# --- 主程式 ---
if __name__ == "__main__":
    # --- !!! 請修改這裡 !!! ---
    # 將 'your_file.txt' 替換成您實際的檔案路徑
    file_to_process = '/home/p76121542/Linux_DATA/synthesis/model/vits/filelists/no_double_phoneme_zh_tw/mixed_5_train_new.txt'
    # -------------------------

    calculate_total_duration(file_to_process)