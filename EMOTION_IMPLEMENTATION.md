# VITS 情感語音合成實作文檔

## 概述

本文檔描述了將 Cross Conditional Attention (CCA) 和 eGeMAPS 特徵萃取整合到 VITS 模型中，以實現情感語音合成的實作細節。

## 改動總結

### 1. 新增模組

#### 1.1 Cross Conditional Attention (CCA)
**檔案:** `attentions.py`
**類別:** `CrossConditionalAttention`

這是一個多頭注意力機制，用於讓文本編碼器關注情感特徵。

**主要功能:**
- 從主特徵（文本編碼）生成 Query
- 從條件特徵（情感/eGeMAPS）生成 Key 和 Value
- 執行跨條件注意力計算
- 支援 mask 機制以處理變長序列

**使用方式:**
```python
cca = CrossConditionalAttention(
    channels=192,           # 主特徵維度
    cond_channels=192,      # 條件特徵維度
    n_heads=4,              # 注意力頭數
    p_dropout=0.1           # Dropout 比率
)

output = cca(
    x=text_features,        # [B, C, T_text]
    cond=emotion_features,  # [B, C_cond, T_emo]
    x_mask=text_mask,       # [B, 1, T_text]
    cond_mask=emo_mask      # [B, 1, T_emo]
)
```

#### 1.2 eGeMAPS Feature Extractor
**檔案:** `egemaps_extractor.py`
**類別:** `eGeMAPS_Extractor`, `eGeMAPS_Encoder`

基於 eGeMAPS v2.0 規範的聲學特徵萃取器。

**萃取的特徵包括:**
1. **MFCC** (13 維): 頻譜特徵
2. **Mel-spectrogram** (80 維): 梅爾頻譜
3. **韻律特徵** (4 維):
   - F0 (基頻)
   - Energy (能量)
   - Spectral Flux (頻譜變化率)
   - Zero Crossing Rate (過零率)

**總特徵維度:** 可配置（默認 88 維）

**使用方式:**
```python
# 特徵萃取器
extractor = eGeMAPS_Extractor(
    sample_rate=22050,
    hop_length=256,
    feature_dim=88
)

# 從音頻提取特徵
features = extractor(waveform)  # [B, 88, T_frames]

# 特徵編碼器（將特徵投影到模型維度）
encoder = eGeMAPS_Encoder(
    feature_dim=88,
    hidden_channels=192,
    out_channels=192
)

encoded_features = encoder(features)  # [B, 192, T_frames]
```

### 2. 修改的模組

#### 2.1 TextEncoder
**檔案:** `models.py`

**新增參數:**
- `use_cca`: bool - 是否使用 CCA
- `emo_channels`: int - 情感特徵維度

**新增功能:**
- 在 Transformer encoder 之後添加 CCA 層
- 可選地將情感特徵融合到文本編碼中

**Forward 簽名:**
```python
def forward(self, x, x_lengths, g=None, emo_feat=None, emo_mask=None):
    # x: 文本輸入
    # emo_feat: 情感特徵（來自 eGeMAPS encoder）
    # emo_mask: 情感特徵的 mask
```

#### 2.2 SynthesizerTrn
**檔案:** `models.py`

**新增參數:**
- `use_cca`: bool - 是否使用 CCA
- `use_egemaps`: bool - 是否使用 eGeMAPS 特徵萃取
- `emo_feature_dim`: int - eGeMAPS 特徵維度
- `sample_rate`: int - 音頻採樣率

**新增模組:**
- `egemaps_extractor`: eGeMAPS 特徵萃取器
- `egemaps_encoder`: eGeMAPS 特徵編碼器

**Forward 簽名:**
```python
def forward(self, x, x_lengths, y, y_lengths,
            sid=None, eid=None, ref_audio=None):
    # ref_audio: 參考音頻用於萃取 eGeMAPS 特徵
```

**Infer 簽名:**
```python
def infer(self, x, x_lengths, sid=None, eid=None, ref_audio=None,
          noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    # ref_audio: 參考音頻用於情感轉移
```

## 使用情境

### 情境 1: 使用 Emotion Embedding（離散情感）

```python
# 配置
config = {
    'n_emotions': 5,        # 5 種情感類別
    'use_cca': False,       # 不使用 CCA
    'use_egemaps': False    # 不使用 eGeMAPS
}

# 訓練
output = model(
    x=text_tokens,
    x_lengths=text_lengths,
    y=mel_spec,
    y_lengths=audio_lengths,
    sid=speaker_ids,
    eid=emotion_ids         # 情感標籤 (0-4)
)

# 推論
audio = model.infer(
    x=text_tokens,
    x_lengths=text_lengths,
    sid=speaker_id,
    eid=emotion_id          # 指定情感
)
```

### 情境 2: 使用 eGeMAPS + CCA（連續情感特徵）

```python
# 配置
config = {
    'use_cca': True,        # 使用 CCA
    'use_egemaps': True,    # 使用 eGeMAPS
    'emo_feature_dim': 88,
    'sample_rate': 22050
}

# 訓練
output = model(
    x=text_tokens,
    x_lengths=text_lengths,
    y=mel_spec,
    y_lengths=audio_lengths,
    sid=speaker_ids,
    ref_audio=reference_audio  # 參考音頻用於萃取情感
)

# 推論（情感轉移）
audio = model.infer(
    x=text_tokens,
    x_lengths=text_lengths,
    sid=speaker_id,
    ref_audio=emotion_reference  # 提供情感參考音頻
)
```

### 情境 3: 混合模式（Emotion Embedding + eGeMAPS）

```python
# 配置
config = {
    'n_emotions': 5,
    'use_cca': True,
    'use_egemaps': True,
    'emo_feature_dim': 88
}

# 訓練
output = model(
    x=text_tokens,
    x_lengths=text_lengths,
    y=mel_spec,
    y_lengths=audio_lengths,
    sid=speaker_ids,
    eid=emotion_ids,           # 粗粒度情感標籤
    ref_audio=reference_audio  # 細粒度情感特徵
)
```

## 配置文件範例

在 `configs/` 目錄下創建新的配置文件：

```json
{
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [8,8,2,2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16,4,4],
    "n_speakers": 1,
    "n_emotions": 5,
    "gin_channels": 256,
    "use_sdp": true,
    "use_cca": true,
    "use_egemaps": true,
    "emo_feature_dim": 88,
    "sample_rate": 22050
  },
  "data": {
    "training_files": "filelists/train.txt",
    "validation_files": "filelists/val.txt",
    "text_cleaners": ["english_cleaners2"],
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 1
  }
}
```

## 訓練數據準備

### 方式 1: 使用情感標籤

數據格式（filelist）:
```
audio_path|text|speaker_id|emotion_id
wavs/1.wav|Hello world|0|1
wavs/2.wav|How are you|0|2
```

### 方式 2: 使用參考音頻

數據格式（filelist）:
```
audio_path|text|speaker_id|ref_audio_path
wavs/1.wav|Hello world|0|emotion_refs/happy_01.wav
wavs/2.wav|How are you|0|emotion_refs/sad_01.wav
```

### 修改 DataLoader

需要修改 `data_utils.py` 中的 `TextAudioSpeakerLoader` 來支援:
1. 讀取 emotion_id 或 ref_audio_path
2. 載入參考音頻
3. 在 batch 中返回額外的數據

## 架構圖

```
訓練流程:
┌─────────────┐
│ Text Input  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌──────────────────┐
│  Text Encoder   │◄────│ Speaker Emb      │
└────────┬────────┘     └──────────────────┘
         │              ┌──────────────────┐
         │              │ Emotion Emb      │
         │              └─────────┬────────┘
         │                        │
         │              ┌─────────▼────────┐
         │              │ Ref Audio        │
         │              └─────────┬────────┘
         │                        │
         │              ┌─────────▼────────┐
         │              │ eGeMAPS Extract  │
         │              └─────────┬────────┘
         │                        │
         │              ┌─────────▼────────┐
         │              │ eGeMAPS Encoder  │
         │              └─────────┬────────┘
         │                        │
         └────────────────────────▼
              Cross Conditional Attention
                        │
                        ▼
              ┌─────────────────┐
              │ Duration Pred   │
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌────────────────┐         ┌────────────────┐
│ Posterior Enc  │         │  Prior (Flow)  │
└────────┬───────┘         └────────┬───────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
            ┌───────────────┐
            │   Decoder     │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  Waveform     │
            └───────────────┘
```

## 下一步工作

1. **數據準備**
   - 收集或標註情感語料庫
   - 準備 filelist 文件
   - 修改 DataLoader

2. **訓練腳本修改**
   - 修改 `train.py` 或 `train_ms.py`
   - 添加 ref_audio 處理邏輯
   - 更新 loss 計算（如需要）

3. **推論腳本**
   - 創建支援情感控制的推論腳本
   - 支援情感轉移（通過參考音頻）
   - 支援情感強度控制

4. **評估**
   - 主觀評估（MOS）
   - 客觀評估（情感分類準確率）
   - A/B 測試

5. **優化**
   - F0 提取改進（使用 CREPE 或 PYIN）
   - eGeMAPS 特徵選擇優化
   - CCA 架構調整

## 參考文獻

1. **VITS**: Kim et al. "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"
2. **eGeMAPS**: Eyben et al. "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing"
3. **EmoSpeech**: (請補充 EmoSpeech 論文引用)

## 疑難排解

### 常見問題

**Q: 模型訓練時出現 NaN loss**
A: 可能是 eGeMAPS 特徵尺度問題，建議添加特徵正規化。

**Q: CCA 沒有效果**
A: 檢查:
1. emo_feat 是否正確傳遞
2. CCA 的權重是否被正確初始化
3. Learning rate 是否合適

**Q: eGeMAPS 提取速度慢**
A: 在訓練時可以預先提取並緩存特徵，而不是即時提取。

**Q: 情感轉移效果不明顯**
A: 考慮:
1. 增加 CCA 層數
2. 調整 attention head 數量
3. 使用更豐富的情感特徵

## 聯絡與貢獻

如有問題或改進建議，請聯繫開發者。

---
最後更新: 2025-12-22
版本: 1.0
