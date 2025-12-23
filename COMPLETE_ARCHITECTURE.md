# VITS 情感語音合成系統 - 完整架構文件

> **最後更新**: 2025-12-22
> **版本**: 2.0
> **專案**: 基於 VITS 的多語者情感語音合成系統 (支援 CCA + eGeMAPS)

---

## 目錄

1. [系統概述](#系統概述)
2. [專案結構](#專案結構)
3. [核心模組詳解](#核心模組詳解)
4. [情感控制機制](#情感控制機制)
5. [訓練流程](#訓練流程)
6. [推論流程](#推論流程)
7. [數據準備](#數據準備)
8. [配置說明](#配置說明)
9. [PromptTTS 風格情感標註](#prompttts-風格情感標註)
10. [常見問題與最佳實踐](#常見問題與最佳實踐)

---

## 系統概述

### 架構特色

本 VITS 系統是一個**端到端**的多語者情感語音合成系統，具備以下特色：

1. **多語者支持**: 支援 1072+ 語者的多語者建模
2. **情感控制**:
   - 離散情感標籤 (Emotion Embedding)
   - 連續情感特徵 (eGeMAPS + CCA)
   - 混合模式 (兩者結合)
3. **多語言支持**: 支援中文、台語、客語、英語、越南語等
4. **靈活的文本前端**: 支援多種音素系統和音調標記

### 技術棧

- **框架**: PyTorch 1.x+
- **模型**: VITS (Variational Inference TTS)
- **情感特徵**: eGeMAPS v2.0
- **注意力機制**: Cross Conditional Attention (CCA)
- **Vocoder**: HiFi-GAN (內建於 VITS)

---

## 專案結構

```
vits/
├── configs/                      # 配置文件目錄
│   ├── ljs_base.json            # 單語者基礎配置
│   ├── vctk_base.json           # 多語者基礎配置
│   └── mixed.json               # 多語者混合配置 (1072 speakers)
│
├── filelists/                   # 訓練數據列表
│   ├── Hakka_hf/               # 客語 (海陸腔) 語料
│   ├── 3646_vad_25_920/        # VAD 處理語料
│   └── [各種語料目錄]/
│       ├── mixed_5_train_new.txt  # 訓練集
│       ├── mixed_5_val_new.txt    # 驗證集
│       └── mixed_5_id.txt         # 語者 ID 映射
│
├── logs/                        # 訓練日誌與 checkpoint
│   └── [model_name]/
│       ├── config.json         # 訓練時使用的配置
│       ├── G_*.pth            # Generator checkpoints
│       └── D_*.pth            # Discriminator checkpoints
│
├── text/                        # 文本處理前端
│   ├── __init__.py             # text_to_sequence 入口
│   ├── symbols.py              # 音素符號定義
│   ├── cleaners.py             # 文本清理器
│   └── [語言處理模組]/
│
├── tools/                       # 工具集
│   ├── phonemes_transformation/ # 音素轉換工具
│   │   └── hakka/              # 客語音素處理
│   ├── asr/                    # ASR 相關工具
│   └── style_gen/              # 風格特徵提取
│
├── monotonic_align/             # MAS 對齊算法
│   ├── core.pyx               # Cython 實現
│   └── setup.py               # 編譯腳本
│
├── models.py                    # 核心模型定義 ⭐
├── attentions.py               # 注意力機制 (含 CCA) ⭐
├── egemaps_extractor.py        # eGeMAPS 特徵提取器 ⭐
├── data_utils.py               # Dataset 與 DataLoader
├── train.py                    # 單語者訓練腳本
├── train_ms.py                 # 多語者訓練腳本 ⭐
├── infer.py                    # 推論腳本
├── losses.py                   # 損失函數
├── modules.py                  # 基礎模組 (ResBlock, WaveNet)
├── commons.py                  # 通用工具函數
├── mel_processing.py           # Mel 頻譜處理
└── utils.py                    # 訓練工具函數

⭐ 表示為情感合成新增或修改的關鍵文件
```

---

## 核心模組詳解

### 1. 模型架構 (models.py)

#### 1.1 SynthesizerTrn - 主模型

這是 VITS 的核心類別，整合了所有子模組。

**關鍵參數**:
```python
SynthesizerTrn(
    n_vocab=128,              # 音素詞彙表大小
    spec_channels=513,        # 頻譜通道數
    segment_size=16,          # 訓練時的音頻片段大小
    inter_channels=192,       # 中間層通道數
    hidden_channels=192,      # 隱藏層通道數
    filter_channels=768,      # Filter 通道數
    n_heads=2,                # 注意力頭數
    n_layers=6,               # Transformer 層數
    kernel_size=3,            # 卷積核大小
    p_dropout=0.1,            # Dropout 比率
    n_speakers=1072,          # 語者數量
    n_emotions=0,             # 情感類別數 (0=不使用)
    gin_channels=256,         # Global conditioning 通道數
    use_sdp=False,            # 是否使用隨機時長預測
    use_cca=True,             # 是否使用 CCA ⭐
    use_egemaps=True,         # 是否使用 eGeMAPS ⭐
    emo_feature_dim=88,       # eGeMAPS 特徵維度 ⭐
    sample_rate=22050,        # 音頻採樣率 ⭐
)
```

**子模組組成**:

1. **TextEncoder** (`self.enc_p`):
   - 將音素序列編碼為隱藏表示
   - 可選地整合 CCA 進行情感條件化

2. **PosteriorEncoder** (`self.enc_q`):
   - 訓練時從真實音頻提取潛在表示
   - 僅用於訓練，推論時不使用

3. **ResidualCouplingFlow** (`self.flow`):
   - Normalizing Flow，學習先驗分佈

4. **Generator/Decoder** (`self.dec`):
   - HiFi-GAN vocoder
   - 將潛在變數解碼為波形

5. **DurationPredictor** (`self.dp`):
   - 預測每個音素的持續時間

6. **StochasticDurationPredictor** (`self.sdp`):
   - 隨機時長預測器 (可選)

7. **eGeMAPS Modules** (新增 ⭐):
   - `self.egemaps_extractor`: 特徵提取器
   - `self.egemaps_encoder`: 特徵編碼器

8. **Embedding Layers**:
   - `self.emb_g`: Speaker embedding (當 n_speakers > 1)
   - `self.emb_e`: Emotion embedding (當 n_emotions > 0)

#### 1.2 訓練前向傳播 (forward)

```python
def forward(self, x, x_lengths, y, y_lengths, sid=None, eid=None, ref_audio=None):
    """
    Args:
        x: [B, T_text] - 音素序列
        x_lengths: [B] - 文本長度
        y: [B, n_mels, T_audio] - Mel 頻譜
        y_lengths: [B] - 音頻長度
        sid: [B] - 語者 ID (可選)
        eid: [B] - 情感 ID (可選)
        ref_audio: [B, T_wav] - 參考音頻 (可選) ⭐

    Returns:
        o: [B, 1, T] - 合成音頻
        l_length: 時長損失
        attn: 對齊矩陣
        ids_slice: 音頻切片索引
        x_mask, y_mask: Masks
        (z, z_p, m_p, logs_p, m_q, logs_q): 潛在變數
    """
```

**執行流程**:

1. **構建 Global Condition** (`g`):
   ```python
   g = emb_g(sid)  # Speaker embedding
   if n_emotions > 0:
       g = g + emb_e(eid)  # 加上 Emotion embedding
   ```

2. **提取 eGeMAPS 特徵** (如果 `use_egemaps=True` 且提供 `ref_audio`):
   ```python
   egemaps_feat = egemaps_extractor(ref_audio)  # [B, 88, T_feat]
   emo_feat = egemaps_encoder(egemaps_feat)     # [B, 192, T_feat]
   ```

3. **Text Encoding + CCA**:
   ```python
   x, m_p, logs_p, x_mask = enc_p(x, x_lengths, g=g,
                                   emo_feat=emo_feat,
                                   emo_mask=emo_mask)
   ```

4. **Posterior Encoding** (從真實音頻):
   ```python
   z, m_q, logs_q, y_mask = enc_q(y, y_lengths, g=g)
   ```

5. **Monotonic Alignment Search** (MAS):
   - 自動學習音素與音頻幀的對齊

6. **Duration Prediction**:
   - 預測每個音素的持續時間

7. **Decoding**:
   ```python
   o = dec(z_slice, g=g)  # 生成音頻波形
   ```

#### 1.3 推論前向傳播 (infer)

```python
def infer(self, x, x_lengths, sid=None, eid=None, ref_audio=None,
          noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    """
    Args:
        x: [B, T_text] - 音素序列
        x_lengths: [B] - 文本長度
        sid: [B] - 語者 ID
        eid: [B] - 情感 ID (可選)
        ref_audio: [B, T_wav] - 情感參考音頻 (可選) ⭐
        noise_scale: Flow 的噪聲尺度
        length_scale: 時長縮放因子 (>1 變慢, <1 變快)
        noise_scale_w: 時長預測的噪聲尺度

    Returns:
        o: [B, 1, T] - 合成音頻
        attn: [B, T_text, T_audio] - 對齊矩陣
    """
```

**關鍵差異**:
- 不使用 PosteriorEncoder
- 使用 Flow 從先驗分佈採樣
- 可通過 `length_scale` 控制語速

---

### 2. Cross Conditional Attention (attentions.py)

#### 2.1 CrossConditionalAttention 類別

這是實現情感特徵融合的核心機制。

```python
class CrossConditionalAttention(nn.Module):
    """
    跨條件注意力機制

    讓文本表示 (Query) 關注情感特徵 (Key, Value)
    """

    def __init__(self,
                 channels,           # 主特徵維度 (文本)
                 cond_channels,      # 條件特徵維度 (情感)
                 n_heads=4,          # 注意力頭數
                 p_dropout=0.1):     # Dropout 比率
        super().__init__()

        self.channels = channels
        self.cond_channels = cond_channels
        self.n_heads = n_heads

        # Query 從主特徵生成
        self.conv_q = nn.Conv1d(channels, channels, 1)

        # Key, Value 從條件特徵生成
        self.conv_k = nn.Conv1d(cond_channels, channels, 1)
        self.conv_v = nn.Conv1d(cond_channels, channels, 1)

        # 輸出投影
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(p_dropout)
```

**前向傳播**:
```python
def forward(self, x, cond, x_mask=None, cond_mask=None):
    """
    Args:
        x: [B, C, T_x] - 主特徵 (文本編碼)
        cond: [B, C_cond, T_cond] - 條件特徵 (情感)
        x_mask: [B, 1, T_x] - 主特徵 mask
        cond_mask: [B, 1, T_cond] - 條件特徵 mask

    Returns:
        output: [B, C, T_x] - 融合後的特徵
    """
    # 1. 計算 Q, K, V
    q = conv_q(x)      # [B, C, T_x]
    k = conv_k(cond)   # [B, C, T_cond]
    v = conv_v(cond)   # [B, C, T_cond]

    # 2. Multi-head attention
    # 重塑為 [B, n_heads, head_dim, T]

    # 3. 計算注意力分數
    attn = softmax(q @ k.T / sqrt(head_dim))

    # 4. 應用注意力
    output = attn @ v

    # 5. 輸出投影
    output = conv_o(output)

    return output
```

#### 2.2 在 TextEncoder 中的使用

```python
class TextEncoder(nn.Module):
    def __init__(self, ..., use_cca=False, emo_channels=0):
        # ...
        if use_cca and emo_channels > 0:
            self.cca = CrossConditionalAttention(
                channels=hidden_channels,
                cond_channels=emo_channels,
                n_heads=n_heads,
                p_dropout=p_dropout
            )
            self.cca_norm = LayerNorm(hidden_channels)

    def forward(self, x, x_lengths, g=None, emo_feat=None, emo_mask=None):
        # 1. Embedding + Transformer
        x = self.encoder(x, x_mask, g=g)

        # 2. 應用 CCA (如果有情感特徵)
        if self.use_cca and emo_feat is not None:
            residual = x
            x_cca = self.cca(x, emo_feat, x_mask=x_mask, cond_mask=emo_mask)
            x = self.cca_norm(residual + x_cca)  # Residual connection

        # 3. 輸出投影
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
```

---

### 3. eGeMAPS 特徵提取器 (egemaps_extractor.py)

#### 3.1 eGeMAPS_Extractor

基於 eGeMAPS v2.0 規範的聲學特徵提取器。

**提取的特徵**:
1. **MFCC** (13 維): Mel-Frequency Cepstral Coefficients
2. **Mel-Spectrogram** (80 維): 梅爾頻譜
3. **韻律特徵** (4 維):
   - F0 (基頻)
   - Energy (能量)
   - Spectral Flux (頻譜變化率)
   - Zero Crossing Rate (過零率)

**總維度**: 13 + 80 + 4 = 97 → 投影到 88 維

```python
class eGeMAPS_Extractor(nn.Module):
    def __init__(self,
                 sample_rate=22050,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=80,
                 f0_min=80,
                 f0_max=600,
                 feature_dim=88):
        super().__init__()

        # Mel-spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(...)

        # MFCC
        self.mfcc = torchaudio.transforms.MFCC(n_mfcc=13, ...)

        # 特徵投影 (97 → 88)
        self.feature_projection = nn.Sequential(
            nn.Linear(13 + 80 + 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim)
        )

    def forward(self, waveform):
        """
        Args:
            waveform: [B, T_wav] - 音頻波形

        Returns:
            features: [B, feature_dim, T_frames] - eGeMAPS 特徵
        """
        # 1. 提取 Mel-spectrogram
        mel = self.mel_spec(waveform)  # [B, n_mels, T_frames]

        # 2. 提取 MFCC
        mfcc = self.mfcc(waveform)     # [B, 13, T_frames]

        # 3. 提取韻律特徵
        f0 = self.extract_f0(waveform)              # [B, T_frames]
        energy = self.extract_energy(mel)           # [B, T_frames]
        flux = self.extract_spectral_flux(mel)      # [B, T_frames]
        zcr = self.extract_zcr(waveform)           # [B, T_frames]

        # 4. 拼接所有特徵
        prosody = torch.stack([f0, energy, flux, zcr], dim=1)  # [B, 4, T_frames]
        all_features = torch.cat([mfcc, mel, prosody], dim=1)  # [B, 97, T_frames]

        # 5. 投影到目標維度
        # [B, 97, T] → [B, T, 97] → projection → [B, T, 88] → [B, 88, T]
        features = all_features.transpose(1, 2)
        features = self.feature_projection(features)
        features = features.transpose(1, 2)

        return features
```

#### 3.2 eGeMAPS_Encoder

將提取的特徵編碼到模型的隱藏空間。

```python
class eGeMAPS_Encoder(nn.Module):
    """
    將 eGeMAPS 特徵編碼為與文本編碼相同維度的表示
    """

    def __init__(self,
                 feature_dim=88,         # eGeMAPS 特徵維度
                 hidden_channels=192,    # 隱藏層維度
                 out_channels=192):      # 輸出維度
        super().__init__()

        self.pre = nn.Conv1d(feature_dim, hidden_channels, 1)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: [B, feature_dim, T] - eGeMAPS 特徵

        Returns:
            output: [B, out_channels, T] - 編碼後的情感特徵
        """
        x = self.pre(x)
        x = self.encoder(x)
        x = self.proj(x)
        return x
```

---

### 4. 數據載入器 (data_utils.py)

#### 4.1 TextAudioSpeakerLoader

多語者數據集類別，負責讀取訓練數據。

**Filelist 格式**:
```
audio_path|speaker_id|language|phoneme_sequence
```

**範例**:
```
/path/to/audio.wav|0|HAK|a33 ph3 oo33 k3 ai33 ...
```

**字段說明**:
- `audio_path`: 音頻文件的絕對路徑
- `speaker_id`: 語者 ID (整數)
- `language`: 語言標識 (ID, EN, TW, ZH, HAK, TZH 等)
- `phoneme_sequence`: 音素序列 (空格分隔)

**支援的語言**:
```python
lang_map = {
    'ID': 0,    # 印尼語
    'EN': 1,    # 英語
    'TW': 2,    # 台語
    'ZH': 3,    # 中文
    'HAK': 4,   # 客語
    'TZH': 5,   # 台灣國語
}
```

#### 4.2 擴展以支援情感數據

如果要支援 PromptTTS 風格的情感標註，需要修改 DataLoader:

**方案 A: 使用情感 ID**
```
audio_path|speaker_id|emotion_id|language|phoneme_sequence
```

**方案 B: 使用參考音頻**
```
audio_path|speaker_id|language|phoneme_sequence|ref_audio_path
```

**方案 C: 使用文本情感描述** (PromptTTS 風格)
```
audio_path|speaker_id|language|phoneme_sequence|emotion_description
```

例如:
```
/path/audio.wav|0|HAK|a33 ph3 oo33|A woman speaks with a joyful and energetic tone
```

---

## 情感控制機制

### 模式 1: 離散情感標籤 (Emotion Embedding)

**適用場景**: 有明確情感類別標註的數據

**配置**:
```json
{
  "model": {
    "n_emotions": 5,      // 5 種情感類別
    "use_cca": false,
    "use_egemaps": false
  }
}
```

**訓練**:
```python
output = model(
    x=text_tokens,
    x_lengths=text_lengths,
    y=mel_spec,
    y_lengths=audio_lengths,
    sid=speaker_ids,
    eid=emotion_ids  # [0, 1, 2, 3, 4]
)
```

**推論**:
```python
audio = model.infer(
    x=text_tokens,
    x_lengths=text_lengths,
    sid=speaker_id,
    eid=emotion_id  # 指定情感類別
)
```

---

### 模式 2: 連續情感特徵 (eGeMAPS + CCA)

**適用場景**: 情感轉移、無明確標籤但有參考音頻

**配置**:
```json
{
  "model": {
    "n_emotions": 0,
    "use_cca": true,
    "use_egemaps": true,
    "emo_feature_dim": 88,
    "sample_rate": 22050
  }
}
```

**訓練**:
```python
output = model(
    x=text_tokens,
    x_lengths=text_lengths,
    y=mel_spec,
    y_lengths=audio_lengths,
    sid=speaker_ids,
    ref_audio=reference_audio  # 從音頻本身提取情感
)
```

**推論** (情感轉移):
```python
# 使用另一段音頻的情感特徵
audio = model.infer(
    x=text_tokens,
    x_lengths=text_lengths,
    sid=speaker_id,
    ref_audio=emotion_reference  # 提供情感參考音頻
)
```

---

### 模式 3: 混合模式

**適用場景**: 粗粒度情感類別 + 細粒度情感特徵

**配置**:
```json
{
  "model": {
    "n_emotions": 5,
    "use_cca": true,
    "use_egemaps": true,
    "emo_feature_dim": 88
  }
}
```

**訓練**:
```python
output = model(
    x=text_tokens,
    x_lengths=text_lengths,
    y=mel_spec,
    y_lengths=audio_lengths,
    sid=speaker_ids,
    eid=emotion_ids,           # 粗粒度 (happy, sad, angry...)
    ref_audio=reference_audio  # 細粒度 (韻律、能量...)
)
```

---

## 訓練流程

### 步驟 1: 環境準備

```bash
# 1. 安裝依賴
pip install torch torchaudio librosa scipy matplotlib tensorboard

# 2. 編譯 Monotonic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
cd ..

# 3. 確認 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 步驟 2: 數據準備

#### 2.1 準備音頻文件

確保所有音頻:
- 採樣率: 22050 Hz
- 格式: 單聲道 WAV
- 音量正規化
- 去除頭尾靜音 (建議使用 VAD)

#### 2.2 準備 Filelist

**格式**:
```
/absolute/path/to/audio1.wav|0|HAK|phoneme sequence here
/absolute/path/to/audio2.wav|0|HAK|another phoneme sequence
```

**分割訓練/驗證集**:
```bash
# 假設有 1000 筆數據
head -900 all_data.txt > mixed_5_train_new.txt
tail -100 all_data.txt > mixed_5_val_new.txt
```

#### 2.3 準備語者 ID 映射

在 `mixed_5_id.txt` 中:
```
0|speaker_name_1
1|speaker_name_2
...
```

### 步驟 3: 配置文件

創建 `configs/my_emotion_config.json`:

```json
{
  "train": {
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 1234,
    "epochs": 10000,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 16,
    "fp16_run": true,
    "lr_decay": 0.999875,
    "segment_size": 4096,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "training_files": "filelists/my_corpus/mixed_5_train_new.txt",
    "validation_files": "filelists/my_corpus/mixed_5_val_new.txt",
    "text_cleaners": ["english_cleaners2"],
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": false,
    "n_speakers": 10,
    "cleaned_text": true
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "gin_channels": 256,
    "use_sdp": false,

    // 情感控制參數 ⭐
    "n_emotions": 0,        // 如果使用 emotion embedding，設為 > 0
    "use_cca": true,        // 啟用 CCA
    "use_egemaps": true,    // 啟用 eGeMAPS
    "emo_feature_dim": 88,  // eGeMAPS 維度
    "sample_rate": 22050
  }
}
```

### 步驟 4: 啟動訓練

```bash
# 多 GPU 訓練 (推薦)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c configs/my_emotion_config.json \
    -m logs/my_emotion_model

# 單 GPU 訓練
CUDA_VISIBLE_DEVICES=0 python train_ms.py \
    -c configs/my_emotion_config.json \
    -m logs/my_emotion_model
```

### 步驟 5: 監控訓練

```bash
# 啟動 TensorBoard
tensorboard --logdir=logs/my_emotion_model --port=6006

# 在瀏覽器打開 http://localhost:6006
```

**關注指標**:
- `train/mel_loss`: Mel 頻譜重建損失
- `train/kl_loss`: KL 散度損失
- `train/duration_loss`: 時長預測損失
- `train/gen_loss`: Generator 損失
- `train/disc_loss`: Discriminator 損失

### 步驟 6: Checkpoint 管理

Checkpoints 保存在 `logs/my_emotion_model/`:
- `G_*.pth`: Generator
- `D_*.pth`: Discriminator

**恢復訓練**:
訓練腳本會自動從最新的 checkpoint 恢復。

---

## 推論流程

### 基本推論腳本

創建 `my_infer.py`:

```python
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence

# 1. 載入配置
hps = utils.get_hparams_from_file("logs/my_emotion_model/config.json")

# 2. 創建模型
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).cuda()

# 3. 載入 checkpoint
_ = net_g.eval()
_ = utils.load_checkpoint("logs/my_emotion_model/G_100000.pth", net_g, None)

# 4. 準備輸入文本
text = "a33 ph3 oo33 k3 ai33"  # 音素序列
text_norm = cleaned_text_to_sequence(text, 'HAK')
text_norm = commons.intersperse(text_norm, 0)
text_norm = torch.LongTensor(text_norm).unsqueeze(0).cuda()
text_lengths = torch.LongTensor([len(text_norm[0])]).cuda()

# 5. 推論
with torch.no_grad():
    audio, attn = net_g.infer(
        x=text_norm,
        x_lengths=text_lengths,
        sid=torch.LongTensor([0]).cuda(),  # Speaker ID
        noise_scale=0.6,
        length_scale=1.0,
        noise_scale_w=0.8
    )

# 6. 保存音頻
from scipy.io.wavfile import write
audio_numpy = audio[0, 0].cpu().numpy()
write("output.wav", hps.data.sampling_rate, audio_numpy)
```

### 情感轉移推論

```python
# 載入參考音頻以提取情感
from utils import load_wav_to_torch

ref_audio_path = "emotion_references/happy_example.wav"
ref_audio, sr = load_wav_to_torch(ref_audio_path)
ref_audio = ref_audio.unsqueeze(0).cuda()

# 推論時提供參考音頻
with torch.no_grad():
    audio, attn = net_g.infer(
        x=text_norm,
        x_lengths=text_lengths,
        sid=torch.LongTensor([0]).cuda(),
        ref_audio=ref_audio,  # 提供情感參考
        noise_scale=0.6,
        length_scale=1.0
    )
```

### 控制語速和情感強度

```python
# 語速控制
audio_slow, _ = net_g.infer(
    ...,
    length_scale=1.5  # >1 變慢, <1 變快
)

# 情感強度 (透過 noise_scale)
audio_expressive, _ = net_g.infer(
    ...,
    noise_scale=0.8,      # 增加表現力
    noise_scale_w=1.0     # 增加時長變化
)
```

---

## PromptTTS 風格情感標註

### 概念

PromptTTS 使用自然語言描述來控制情感和風格，例如:
```
"A woman speaks with a happy and energetic tone"
"A man speaks slowly with sadness"
```

### 實作策略

#### 方案 1: 文本編碼器 (推薦)

使用預訓練的語言模型 (如 BERT) 將情感描述編碼為向量。

**架構修改**:

1. **新增 Prompt Encoder**:
```python
from transformers import AutoModel, AutoTokenizer

class PromptEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_channels=192):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(768, hidden_channels)  # BERT hidden → model hidden

    def forward(self, prompts):
        """
        Args:
            prompts: List[str] - 情感描述文本列表
        Returns:
            prompt_emb: [B, hidden_channels, 1] - 編碼後的 prompt
        """
        # Tokenize
        inputs = self.tokenizer(prompts, return_tensors='pt',
                               padding=True, truncation=True)
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.bert(**inputs)
            pooled = outputs.pooler_output  # [B, 768]

        # Project
        prompt_emb = self.proj(pooled)  # [B, hidden_channels]
        prompt_emb = prompt_emb.unsqueeze(-1)  # [B, hidden_channels, 1]

        return prompt_emb
```

2. **整合到 SynthesizerTrn**:
```python
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., use_prompt=False):
        # ...
        if use_prompt:
            self.prompt_encoder = PromptEncoder()

    def forward(self, x, x_lengths, y, y_lengths,
                sid=None, emotion_prompts=None):
        # ...
        # 將 prompt 編碼後加到 g
        if self.use_prompt and emotion_prompts is not None:
            prompt_emb = self.prompt_encoder(emotion_prompts)
            if g is None:
                g = prompt_emb
            else:
                g = g + prompt_emb
        # ...
```

3. **數據格式**:
```
audio_path|speaker_id|language|phoneme|emotion_description
/path/audio.wav|0|HAK|a33 ph3|A woman speaks happily
```

4. **DataLoader 修改**:
```python
class TextAudioSpeakerEmotionLoader(torch.utils.data.Dataset):
    def __getitem__(self, index):
        audiopath, sid, lang, phoneme, emotion_desc = self.data[index]
        # ...
        return text, spec, wav, sid, emotion_desc
```

#### 方案 2: CLAP (Contrastive Language-Audio Pretraining)

使用 CLAP 模型對齊文本描述和音頻特徵。

**優點**:
- 直接學習文本-音頻對應
- 不需要離散情感標籤

**實作**:
```python
from transformers import ClapModel, ClapProcessor

class CLAPEmotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.proj = nn.Linear(512, 192)  # CLAP → model hidden

    def forward(self, audio=None, text=None):
        """
        可以從音頻或文本提取情感特徵
        """
        if audio is not None:
            inputs = self.processor(audios=audio, return_tensors="pt", sampling_rate=48000)
            audio_embeds = self.model.get_audio_features(**inputs)
            return self.proj(audio_embeds)

        if text is not None:
            inputs = self.processor(text=text, return_tensors="pt")
            text_embeds = self.model.get_text_features(**inputs)
            return self.proj(text_embeds)
```

### 情感描述模板

為了保持一致性，建議使用結構化的描述模板:

```python
EMOTION_TEMPLATES = {
    'happy': "A {gender} speaks with a joyful and energetic tone",
    'sad': "A {gender} speaks with a sad and melancholic tone",
    'angry': "A {gender} speaks with an angry and intense tone",
    'neutral': "A {gender} speaks in a calm and neutral tone",
    'excited': "A {gender} speaks with excitement and enthusiasm",
}

def generate_emotion_prompt(emotion, gender='woman',
                           intensity='moderate',
                           speed='normal'):
    base = EMOTION_TEMPLATES[emotion].format(gender=gender)

    if intensity == 'strong':
        base += ", with strong emphasis"
    elif intensity == 'subtle':
        base += ", with subtle expression"

    if speed == 'slow':
        base += ", speaking slowly"
    elif speed == 'fast':
        base += ", speaking quickly"

    return base

# 使用範例
prompt = generate_emotion_prompt('happy', gender='woman', intensity='strong')
# "A woman speaks with a joyful and energetic tone, with strong emphasis"
```

### 訓練數據準備

如果沒有人工標註的情感描述，可以使用以下策略:

#### 策略 1: 基於規則生成

根據音頻的聲學特徵自動生成描述:

```python
def auto_generate_prompt(audio_path):
    # 1. 提取聲學特徵
    y, sr = librosa.load(audio_path, sr=22050)

    # 2. 分析特徵
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=600)
    f0_mean = np.nanmean(f0)
    f0_std = np.nanstd(f0)
    energy = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(energy)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # 3. 生成描述
    gender = 'woman' if f0_mean > 180 else 'man'

    if f0_std > 50:
        emotion = 'excited' if energy_mean > 0.05 else 'expressive'
    elif f0_std < 20:
        emotion = 'calm' if energy_mean < 0.03 else 'neutral'
    else:
        emotion = 'normal'

    speed = 'slowly' if tempo < 100 else ('quickly' if tempo > 140 else 'normally')

    return f"A {gender} speaks {emotion}ly, {speed}"
```

#### 策略 2: 使用情感識別模型

使用預訓練的情感識別模型自動標註:

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
emotion_processor = Wav2Vec2Processor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

def predict_emotion(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = emotion_processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = emotion_model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    return emotion_labels[predicted_id]
```

---

## 配置說明

### 完整配置文件解析

```json
{
  "train": {
    "log_interval": 200,        // 每 N 步記錄一次日誌
    "eval_interval": 1000,      // 每 N 步進行一次驗證
    "seed": 1234,               // 隨機種子
    "epochs": 10000,            // 訓練總輪數
    "learning_rate": 2e-4,      // 學習率
    "betas": [0.8, 0.99],       // Adam optimizer betas
    "eps": 1e-9,                // Adam epsilon
    "batch_size": 16,           // Batch size
    "fp16_run": true,           // 是否使用混合精度訓練
    "lr_decay": 0.999875,       // 學習率衰減係數
    "segment_size": 4096,       // 音頻片段大小 (樣本點)
    "init_lr_ratio": 1,         // 初始學習率比率
    "warmup_epochs": 0,         // Warmup 輪數
    "c_mel": 45,                // Mel loss 權重
    "c_kl": 1.0                 // KL loss 權重
  },

  "data": {
    "training_files": "filelists/train.txt",
    "validation_files": "filelists/val.txt",
    "text_cleaners": ["english_cleaners2"],  // 文本清理器
    "max_wav_value": 32768.0,   // 音頻最大值 (16-bit)
    "sampling_rate": 22050,     // 採樣率
    "filter_length": 1024,      // FFT 大小
    "hop_length": 256,          // Hop size
    "win_length": 1024,         // 窗口大小
    "n_mel_channels": 80,       // Mel 頻道數
    "mel_fmin": 0.0,            // Mel 最低頻率
    "mel_fmax": null,           // Mel 最高頻率 (null = sr/2)
    "add_blank": false,         // 是否在音素間添加 blank
    "n_speakers": 1,            // 語者數量
    "cleaned_text": true        // 文本是否已清理
  },

  "model": {
    "inter_channels": 192,      // 中間層通道數
    "hidden_channels": 192,     // 隱藏層通道數
    "filter_channels": 768,     // Filter 通道數
    "n_heads": 2,               // 注意力頭數
    "n_layers": 6,              // Transformer 層數
    "kernel_size": 3,           // 卷積核大小
    "p_dropout": 0.1,           // Dropout 比率
    "resblock": "1",            // ResBlock 類型
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [8, 8, 2, 2],          // 上採樣倍率
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3,            // Posterior encoder 層數
    "use_spectral_norm": false, // 是否使用 spectral normalization
    "gin_channels": 256,        // Global conditioning 通道數
    "use_sdp": false,           // 是否使用隨機時長預測

    // 情感控制參數
    "n_emotions": 0,            // 情感類別數 (0=不使用)
    "use_cca": true,            // 是否使用 CCA
    "use_egemaps": true,        // 是否使用 eGeMAPS
    "emo_feature_dim": 88,      // eGeMAPS 特徵維度
    "sample_rate": 22050        // 音頻採樣率 (for eGeMAPS)
  }
}
```

### 關鍵參數調整建議

#### 模型大小 vs 質量

| 參數 | 小模型 | 中模型 | 大模型 |
|------|--------|--------|--------|
| `hidden_channels` | 128 | 192 | 256 |
| `filter_channels` | 512 | 768 | 1024 |
| `n_heads` | 2 | 2 | 4 |
| `n_layers` | 4 | 6 | 8 |
| VRAM 需求 | ~8GB | ~12GB | ~20GB |

#### 訓練速度 vs 穩定性

```json
{
  // 快速訓練 (可能不穩定)
  "batch_size": 32,
  "learning_rate": 4e-4,
  "fp16_run": true,

  // 穩定訓練 (較慢)
  "batch_size": 16,
  "learning_rate": 2e-4,
  "fp16_run": false
}
```

#### 情感控制強度

```json
{
  // 弱情感控制
  "n_heads": 2,           // CCA 注意力頭數
  "use_cca": true,
  "use_egemaps": true,

  // 強情感控制
  "n_heads": 4,           // 更多注意力頭
  "emo_feature_dim": 128, // 更豐富的特徵
  // 可考慮在 TextEncoder 中添加多層 CCA
}
```

---

## 常見問題與最佳實踐

### Q1: 訓練時 Loss 出現 NaN

**可能原因**:
1. 學習率過高
2. eGeMAPS 特徵尺度問題
3. Gradient explosion

**解決方法**:
```python
# 1. 降低學習率
"learning_rate": 1e-4  # 從 2e-4 降低

# 2. 在 eGeMAPS_Extractor 中添加正規化
class eGeMAPS_Extractor(nn.Module):
    def forward(self, waveform):
        features = ...
        # 添加 L2 正規化
        features = F.normalize(features, p=2, dim=1)
        return features

# 3. Gradient clipping
torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=5.0)
```

### Q2: CCA 沒有效果

**檢查清單**:
1. `emo_feat` 是否正確傳遞
2. CCA 權重是否被正確初始化
3. CCA 的輸出是否被使用

**Debug 方法**:
```python
# 在 TextEncoder.forward 中添加
if self.use_cca and emo_feat is not None:
    print(f"[DEBUG] emo_feat shape: {emo_feat.shape}")
    print(f"[DEBUG] emo_feat mean: {emo_feat.mean()}, std: {emo_feat.std()}")

    x_cca = self.cca(x, emo_feat, x_mask=x_mask, cond_mask=emo_mask)
    print(f"[DEBUG] x_cca shape: {x_cca.shape}")
    print(f"[DEBUG] x_cca mean: {x_cca.mean()}, std: {x_cca.std()}")
```

### Q3: 情感轉移效果不明顯

**改進策略**:

1. **增加 CCA 層數**:
```python
class TextEncoder(nn.Module):
    def __init__(self, ..., use_cca=False, emo_channels=0, n_cca_layers=2):
        # ...
        if use_cca and emo_channels > 0:
            self.cca_layers = nn.ModuleList([
                CrossConditionalAttention(...) for _ in range(n_cca_layers)
            ])
            self.cca_norms = nn.ModuleList([
                LayerNorm(...) for _ in range(n_cca_layers)
            ])

    def forward(self, x, x_lengths, g=None, emo_feat=None, emo_mask=None):
        x = self.encoder(x, x_mask, g=g)

        if self.use_cca and emo_feat is not None:
            for cca, norm in zip(self.cca_layers, self.cca_norms):
                residual = x
                x_cca = cca(x, emo_feat, x_mask=x_mask, cond_mask=emo_mask)
                x = norm(residual + x_cca)

        return x, m, logs, x_mask
```

2. **使用更豐富的情感特徵**:
```python
# 增加 eGeMAPS 特徵維度
"emo_feature_dim": 128  # 從 88 增加到 128

# 或使用其他特徵提取器
# 例如: Wav2Vec2, HuBERT
```

3. **調整注意力頭數**:
```python
CrossConditionalAttention(
    channels=192,
    cond_channels=192,
    n_heads=8,  # 從 4 增加到 8
    p_dropout=0.1
)
```

### Q4: eGeMAPS 提取速度慢

**優化方法**:

1. **預先提取特徵**:
```python
# 預處理腳本
import torch
from egemaps_extractor import eGeMAPS_Extractor

extractor = eGeMAPS_Extractor()

for audio_path in audio_paths:
    audio, sr = load_audio(audio_path)
    features = extractor(audio)

    # 保存特徵
    feature_path = audio_path.replace('.wav', '.egemaps.pt')
    torch.save(features, feature_path)
```

2. **修改 DataLoader**:
```python
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    def get_audio(self, filename):
        # ...
        # 載入預計算的 eGeMAPS 特徵
        egemaps_path = filename.replace('.wav', '.egemaps.pt')
        if os.path.exists(egemaps_path):
            egemaps_feat = torch.load(egemaps_path)
        else:
            # Fallback: 即時提取
            egemaps_feat = self.egemaps_extractor(audio)

        return spec, audio, egemaps_feat
```

### Q5: 多語者訓練時某些語者效果差

**可能原因**:
1. 數據不平衡
2. 語者 embedding 維度不足
3. 某些語者的數據質量差

**解決方法**:

1. **數據平衡**:
```python
# 在 DistributedBucketSampler 中添加語者平衡
class BalancedSpeakerSampler(DistributedBucketSampler):
    def __init__(self, dataset, ..., balance_speakers=True):
        # 統計每個語者的樣本數
        speaker_counts = {}
        for item in dataset:
            sid = item['speaker_id']
            speaker_counts[sid] = speaker_counts.get(sid, 0) + 1

        # 計算採樣權重
        if balance_speakers:
            weights = [1.0 / speaker_counts[item['speaker_id']]
                      for item in dataset]
        # ...
```

2. **增加 Speaker Embedding 維度**:
```python
"gin_channels": 512  // 從 256 增加到 512
```

3. **語者特定的 Fine-tuning**:
```python
# 對效果差的語者進行額外訓練
# 凍結大部分參數,只訓練 speaker embedding
for param in net_g.parameters():
    param.requires_grad = False
net_g.emb_g.weight.requires_grad = True
```

### Q6: 合成語音的韻律不自然

**改進建議**:

1. **啟用 Stochastic Duration Predictor**:
```json
{
  "model": {
    "use_sdp": true  // 啟用 SDP
  }
}
```

2. **調整推論時的 noise scales**:
```python
audio, attn = net_g.infer(
    ...,
    noise_scale=0.8,      # 增加變化性 (0.6 → 0.8)
    noise_scale_w=1.0,    # 增加時長變化 (0.8 → 1.0)
    length_scale=1.0
)
```

3. **使用更長的訓練音頻片段**:
```json
{
  "train": {
    "segment_size": 8192  // 從 4096 增加到 8192
  }
}
```

### Q7: 如何評估情感控制效果

**評估方法**:

1. **客觀評估**:
```python
# 使用情感識別模型
from transformers import Wav2Vec2ForSequenceClassification

def evaluate_emotion_accuracy(generated_audio, target_emotion):
    emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(...)

    # 預測情感
    predicted_emotion = emotion_model(generated_audio)

    # 計算準確率
    accuracy = (predicted_emotion == target_emotion).float().mean()
    return accuracy
```

2. **主觀評估 (MOS)**:
```python
# 準備評估樣本
evaluation_samples = [
    ('neutral', 'sample_neutral.wav'),
    ('happy', 'sample_happy.wav'),
    ('sad', 'sample_sad.wav'),
]

# 使用 Amazon MTurk 或內部評估
# 問題: "這段語音聽起來有多自然? (1-5分)"
# 問題: "這段語音的情感是否符合預期? (是/否)"
```

3. **A/B 測試**:
```python
# 比較有無情感控制的差異
samples_A = generate_samples(use_emotion=False)
samples_B = generate_samples(use_emotion=True)

# 評估者選擇更自然/更有表現力的版本
```

---

## 附錄

### A. 符號表 (symbols.py)

當前系統支援的音素符號:

```python
# text/symbols.py
_pad = '_'
_punctuation = ';:,.!?¡¿—…-–"«»"" '
_tone = '0123456789'

# 客語音素範例
_hakka_initials = ['p', 'ph', 't', 'th', 'k', 'kh', 'ts', 'tsh', ...]
_hakka_finals = ['a', 'e', 'i', 'o', 'u', 'ai', 'au', 'oi', ...]

symbols = [_pad] + list(_punctuation) + list(_tone) + ...
```

### B. 預訓練模型下載

待補充...

### C. 引用

如果本專案對您的研究有幫助,請引用:

```bibtex
@inproceedings{kim2021vits,
  title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={ICML},
  year={2021}
}

@article{eyben2015geneva,
  title={The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and affective computing},
  author={Eyben, Florian and Scherer, Klaus R and Schuller, Bj{\"o}rn W and others},
  journal={IEEE transactions on affective computing},
  year={2015}
}
```

### D. 聯絡資訊

- 專案維護者: [您的名字/團隊]
- Email: [聯絡信箱]
- GitHub Issues: [專案 GitHub 連結]

---

**文件結束**

最後更新: 2025-12-22
