# VITS 語音合成使用指南

## 快速開始

### 環境啟動
```bash
conda activate vits
cd /home/p76131482/Linux_DATA/synthesis/model/vits
```

---

## 📋 列出可用模型

```bash
python infer.py --list-models
```

目前支援的模型：
| 語言 | 模型名稱 | 說明 |
|------|----------|------|
| 客家語 | `hakka_hf` | 海陸腔 (Female) |
| 客家語 | `hakka_hm` | 海陸腔 (Male) |
| 客家語 | `hakka_xf` | 四縣腔 (Female) |
| 客家語 | `hakka_xm` | 四縣腔 (Male) |
| 台語 | `tw_new_2` | 台語新版 v2 |
| 台語 | `retraintw` | 台語 Retrain |
| 英語 | `en_0111` | 英語模型 |
| 越南語 | `vietnamese` | 越南語 |

---

## 🎤 單句合成

### 基本用法
```bash
python infer.py --text "音素序列" --sid 說話人ID
```

### 範例
```bash
# 使用預設模型 (hakka_hf)
python infer.py --text "sil l3 oo31 th3 ai38 sil" --sid 0

# 指定模型
python infer.py --model hakka_hm --text "sil l3 oo31 th3 ai38 sil" --sid 0

# 指定輸出路徑
python infer.py --model hakka_hf --text "sil tsh3 iu32 tsh3 in35 sil" --sid 0 --output ./output.wav
```

### 調整合成參數
```bash
python infer.py --text "..." --sid 0 \
  --noise-scale 0.3 \
  --noise-scale-w 0.3 \
  --length-scale 1.4    # 值越大語速越慢
```

---

## 📂 批次合成

### 輸入檔案格式
每行：`檔名|音素序列`
```
audio_001|sil l3 oo31 th3 ai38 sil
audio_002|sil k3 im38 p3 u38 sil
```

### 執行批次合成
```bash
python infer.py --model hakka_hf --batch input.txt --output-dir ./gen_audio/batch --sid 0
```

### 不自動加 sil
```bash
python infer.py --batch input.txt --output-dir ./output --sid 0 --no-sil
```

---

## ➕ 新增模型配置

編輯 `inference_config.yaml`：

```yaml
models:
  my_new_model:
    name: "我的新模型"
    config: "logs/my_model/config.json"
    checkpoint: "logs/my_model/G_100000.pth"
    speaker_file: "filelists/my_model/mixed_5_id.txt"
    lang_phones: "filelists/my_model/lang_phones.txt"
    default_language: "TW"   # ZH/TW/HAK/EN/VI
```

設為預設模型：
```yaml
default_model: my_new_model
```

---

## 📁 檔案說明

| 檔案 | 用途 |
|------|------|
| `infer.py` | 新版 CLI（推薦使用）|
| `infer_legacy.py` | 舊版推論腳本（備份）|
| `vits_inferencer.py` | VITSInferencer 類別 |
| `inference_config.yaml` | 模型配置檔 |

---

## 🔧 完整參數列表

```bash
python infer.py --help
```

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--model`, `-m` | 模型名稱 | 預設模型 |
| `--text`, `-t` | 合成文字（音素） | 必填 |
| `--sid` | 說話人 ID | 0 |
| `--lang` | 語言標籤 | 模型預設 |
| `--output`, `-o` | 輸出路徑 | 自動產生 |
| `--batch`, `-b` | 批次輸入檔 | - |
| `--output-dir` | 批次輸出目錄 | ./gen_audio/batch |
| `--no-sil` | 不自動加 sil | False |
| `--noise-scale` | 噪音比例 | 0.3 |
| `--length-scale` | 長度比例 | 1.4 |
