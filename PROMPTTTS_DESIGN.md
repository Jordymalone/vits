# PromptTTS 風格情感控制 - 設計文件

## 目標

用自然語言描述來控制情感表現，例如:
```
"A woman speaks with a happy and energetic tone"
"Speaking slowly with sadness and low energy"
```

**取代** reference audio 的情感提取方式。

---

## 為什麼選擇 PromptTTS 風格？

### 優勢

1. **無需參考音頻**: 推論時不需要額外的音頻文件
2. **更直觀**: 自然語言描述比選擇參考音頻更容易
3. **靈活性高**: 可以組合多種情感屬性
4. **可解釋性**: 清楚知道模型應該生成什麼樣的情感

### 挑戰

1. **文本-情感對齊**: 如何讓文本描述準確映射到聲學特徵
2. **訓練數據**: 需要大量帶有情感描述的語料
3. **特徵空間**: 文本 embedding 如何與聲學特徵空間對齊

---

## 架構設計方案對比

### 方案 A: BERT-based Prompt Encoder (推薦 ⭐)

**優點**:
- 實作簡單
- 預訓練模型效果好
- 訓練穩定

**缺點**:
- 需要大量情感描述文本
- 可能存在 domain gap (文本 vs 音頻)

**架構**:
```
Prompt Text → BERT Encoder → MLP → Emotion Embedding → 加到 g (speaker emb)
                                                      ↓
                                              或用 CCA 注入到 Text Encoder
```

**實作複雜度**: ★★☆☆☆ (中低)

---

### 方案 B: CLAP-based Prompt Encoder (最佳 ⭐⭐⭐)

**優點**:
- 文本和音頻天然對齊 (CLAP 就是為此設計的)
- 可以同時支援文本描述和參考音頻
- 零樣本能力強

**缺點**:
- 需要額外的 CLAP 模型
- 推論時略慢
- CLAP 模型較大 (~300MB)

**架構**:
```
                    ┌─→ CLAP Text Encoder ─┐
Prompt Text ────────┤                       ├─→ Shared Embedding Space ─→ MLP ─→ Emotion Features
或 Reference Audio ─┘                       │
                    └─→ CLAP Audio Encoder ─┘
```

**實作複雜度**: ★★★☆☆ (中)

---

### 方案 C: Dual-Mode Encoder (靈活 ⭐⭐)

**優點**:
- 同時支援文本和音頻
- 可以在推論時自由切換
- 訓練時可以互相增強

**缺點**:
- 實作較複雜
- 需要設計對齊策略

**架構**:
```
                ┌─→ BERT ─→ MLP ─┐
Prompt Text ────┤                 ├─→ Unified Emotion Space ─→ CCA ─→ Text Encoder
                └─────────────────┘
                         ↕ (alignment loss)
                ┌────────────────┐
Reference Audio ┤                │
                └→ F0+Energy ─→ MLP
```

**實作複雜度**: ★★★★☆ (高)

---

## 推薦方案：CLAP-based (方案 B)

### 為什麼選擇 CLAP？

1. **天然的跨模態對齊**:
   - CLAP (Contrastive Language-Audio Pretraining) 已經學會了文本和音頻的對應關係
   - 不需要額外的對齊訓練

2. **支援多種輸入**:
   - 訓練時可以用音頻
   - 推論時可以用文本描述
   - 甚至可以同時使用 (multi-modal fusion)

3. **零樣本能力**:
   - CLAP 可以理解沒見過的情感描述
   - 例如: "speaking with a mix of happiness and excitement"

---

## 詳細實作設計

### 架構圖

```
┌──────────────────────────────────────────────────────────────────┐
│                         Training Phase                            │
└──────────────────────────────────────────────────────────────────┘

Input Text ──→ Text Encoder ──→ ... ──→ VITS Model
                    ↑
                    │ (concat or add)
                    │
Emotion Prompt ──→ CLAP Text Encoder ──→ MLP ──→ Emotion Embedding
  (text)                                             │
                                                     ├─→ 加到 g
                                                     │   或
                                                     └─→ CCA 注入


┌──────────────────────────────────────────────────────────────────┐
│                        Inference Phase                            │
└──────────────────────────────────────────────────────────────────┘

Input Text ──→ Text Encoder ──→ ... ──→ VITS Model ──→ Waveform
                    ↑
                    │
Emotion Prompt ──→ CLAP Text Encoder ──→ MLP ──→ Emotion Embedding
  "A woman speaks
   happily"
```

### 核心組件

#### 1. CLAP Emotion Encoder

```python
from transformers import ClapModel, ClapProcessor

class CLAPEmotionEncoder(nn.Module):
    def __init__(
        self,
        clap_model_name="laion/clap-htsat-unfused",
        hidden_channels=192,
        freeze_clap=True  # 是否凍結 CLAP 權重
    ):
        super().__init__()

        # 載入預訓練 CLAP
        self.processor = ClapProcessor.from_pretrained(clap_model_name)
        self.clap_model = ClapModel.from_pretrained(clap_model_name)

        # 凍結 CLAP 權重 (可選)
        if freeze_clap:
            for param in self.clap_model.parameters():
                param.requires_grad = False

        # 投影層: CLAP embedding (512) → model hidden (192)
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_channels)
        )

    def encode_text(self, texts):
        """
        從文本提取情感 embedding

        Args:
            texts: List[str] - 情感描述
        Returns:
            emotion_emb: [B, hidden_channels, 1]
        """
        # Tokenize
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.clap_model.device) for k, v in inputs.items()}

        # CLAP text encoding
        with torch.no_grad():
            text_embeds = self.clap_model.get_text_features(**inputs)
            # [B, 512]

        # Project to model dimension
        emotion_emb = self.projection(text_embeds)  # [B, hidden_channels]
        emotion_emb = emotion_emb.unsqueeze(-1)     # [B, hidden_channels, 1]

        return emotion_emb

    def encode_audio(self, audio):
        """
        從音頻提取情感 embedding (訓練時使用)

        Args:
            audio: [B, T] waveform at 48kHz
        Returns:
            emotion_emb: [B, hidden_channels, 1]
        """
        # Resample to 48kHz if needed (CLAP 要求)
        # ...

        inputs = self.processor(
            audios=audio,
            return_tensors="pt",
            sampling_rate=48000
        )
        inputs = {k: v.to(self.clap_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_embeds = self.clap_model.get_audio_features(**inputs)
            # [B, 512]

        emotion_emb = self.projection(audio_embeds)
        emotion_emb = emotion_emb.unsqueeze(-1)

        return emotion_emb

    def forward(self, texts=None, audio=None):
        """
        支援文本或音頻輸入
        """
        if texts is not None:
            return self.encode_text(texts)
        elif audio is not None:
            return self.encode_audio(audio)
        else:
            raise ValueError("Must provide either texts or audio")
```

#### 2. 整合到 SynthesizerTrn

```python
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., use_clap_emotion=False):
        super().__init__()

        # ... (原有的初始化)

        if use_clap_emotion:
            self.emotion_encoder = CLAPEmotionEncoder(
                hidden_channels=gin_channels  # 與 speaker emb 相同維度
            )

    def forward(self, x, x_lengths, y, y_lengths,
                sid=None, emotion_prompts=None):
        """
        Args:
            emotion_prompts: List[str] - 情感描述文本
                例如: ["A woman speaks happily", "A man speaks with anger"]
        """
        g = None

        # Speaker embedding
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)

        # Emotion embedding from text prompt
        if hasattr(self, 'emotion_encoder') and emotion_prompts is not None:
            emotion_emb = self.emotion_encoder(texts=emotion_prompts)

            if g is None:
                g = emotion_emb
            else:
                # 兩種融合方式:
                # 方式 1: 簡單相加
                g = g + emotion_emb

                # 方式 2: 加權融合 (可學習)
                # g = self.speaker_weight * g + self.emotion_weight * emotion_emb

        # 後續與原來相同
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        # ...

    def infer(self, x, x_lengths, sid=None, emotion_prompt=None, ...):
        """
        推論時使用文本描述

        Args:
            emotion_prompt: str - 單個情感描述
                例如: "A woman speaks with joy and energy"
        """
        g = None

        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)

        # 從文本提取情感
        if hasattr(self, 'emotion_encoder') and emotion_prompt is not None:
            emotion_emb = self.emotion_encoder(texts=[emotion_prompt])

            if g is None:
                g = emotion_emb
            else:
                g = g + emotion_emb

        # 後續與原來相同
        # ...
```

---

## 訓練數據準備策略

### 問題: 如何獲得大量情感描述？

#### 策略 1: 自動生成 (規則 + 聲學分析)

```python
def generate_emotion_prompt(audio_path):
    """
    基於聲學特徵自動生成情感描述
    """
    # 1. 提取聲學特徵
    f0, energy = extract_minimal_features(audio_path)

    # 2. 分析特徵
    f0_mean = f0[f0 > 0].mean()
    f0_std = f0[f0 > 0].std()
    energy_mean = energy.mean()
    tempo = estimate_tempo(audio_path)

    # 3. 生成描述
    components = []

    # Gender (based on F0)
    gender = "woman" if f0_mean > 180 else "man"
    components.append(f"A {gender} speaks")

    # Emotion (based on F0 variance and energy)
    if f0_std > 50 and energy_mean > 0.05:
        emotion = "with excitement and energy"
    elif f0_std < 20 and energy_mean < 0.03:
        emotion = "calmly and softly"
    elif energy_mean > 0.06:
        emotion = "loudly and assertively"
    else:
        emotion = "in a neutral tone"
    components.append(emotion)

    # Speed (based on tempo)
    if tempo > 140:
        components.append("speaking quickly")
    elif tempo < 100:
        components.append("speaking slowly")

    return ", ".join(components)

# 範例輸出:
# "A woman speaks, with excitement and energy, speaking quickly"
```

#### 策略 2: 使用情感識別模型 + 模板

```python
from transformers import pipeline

# 載入情感識別模型
emotion_classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

EMOTION_TEMPLATES = {
    'happy': [
        "speaking with a happy and cheerful tone",
        "speaking joyfully and with enthusiasm",
        "in a bright and positive manner"
    ],
    'sad': [
        "speaking with sadness and melancholy",
        "in a sorrowful and dejected tone",
        "speaking sadly and quietly"
    ],
    'angry': [
        "speaking angrily and intensely",
        "with frustration and irritation",
        "in an aggressive and harsh manner"
    ],
    'neutral': [
        "speaking in a calm and neutral tone",
        "with a steady and composed manner",
        "in a matter-of-fact way"
    ],
    # ... 更多情感
}

def generate_prompt_from_audio(audio_path):
    # 1. 預測情感
    result = emotion_classifier(audio_path)
    emotion = result[0]['label'].lower()
    confidence = result[0]['score']

    # 2. 分析其他屬性
    gender = detect_gender(audio_path)
    speed = analyze_speaking_rate(audio_path)

    # 3. 組合模板
    emotion_desc = random.choice(EMOTION_TEMPLATES.get(emotion, ['normally']))

    prompt = f"A {gender} speaks {emotion_desc}"

    if speed == 'fast':
        prompt += ", speaking quickly"
    elif speed == 'slow':
        prompt += ", speaking slowly"

    return prompt, emotion, confidence

# 範例輸出:
# ("A woman speaks with a happy and cheerful tone, speaking quickly", 'happy', 0.95)
```

#### 策略 3: 人工標註 (高質量但成本高)

**標註指南**:
```
為每段音頻提供自然語言描述，包含:
1. 說話者性別 (A woman/A man)
2. 主要情感 (happy, sad, angry, neutral, surprised, fearful)
3. 情感強度 (slightly, moderately, very, extremely)
4. 語速 (slowly, normally, quickly)
5. 音量 (quietly, normally, loudly)
6. (可選) 其他特徵

範例:
- "A woman speaks with moderate happiness, speaking at a normal pace"
- "A man speaks very angrily and loudly, speaking quickly"
- "A woman speaks slightly sadly and quietly, speaking slowly"
```

**標註工具**:
可以使用 Label Studio 或 Prodigy 建立標註界面。

#### 策略 4: 混合方法 (推薦 ⭐)

```python
def prepare_training_data(audio_paths):
    """
    混合自動生成和人工校正
    """
    results = []

    for audio_path in tqdm(audio_paths):
        # 1. 自動生成初始描述
        auto_prompt = generate_emotion_prompt(audio_path)

        # 2. 使用情感模型驗證
        _, emotion, confidence = generate_prompt_from_audio(audio_path)

        # 3. 只有高置信度的自動接受
        if confidence > 0.85:
            final_prompt = auto_prompt
            needs_review = False
        else:
            final_prompt = auto_prompt
            needs_review = True  # 標記為需要人工審核

        results.append({
            'audio_path': audio_path,
            'prompt': final_prompt,
            'needs_review': needs_review,
            'auto_confidence': confidence
        })

    # 4. 輸出需要審核的樣本
    to_review = [r for r in results if r['needs_review']]
    print(f"Total: {len(results)}, Need review: {len(to_review)}")

    return results, to_review
```

---

## Filelist 格式

### 新格式

```
audio_path|speaker_id|language|phoneme_sequence|emotion_prompt
```

### 範例

```
/path/audio1.wav|0|HAK|a33 ph3 oo33|A woman speaks with happiness and energy
/path/audio2.wav|0|HAK|t3 ak38 am35|A woman speaks calmly in a neutral tone
/path/audio3.wav|1|ZH|ni3 hao3|A man speaks with slight anger, speaking quickly
```

---

## 實作步驟 (循序漸進)

### Phase 1: 基礎實作 (1-2 週)

1. ✅ 實作 `CLAPEmotionEncoder`
2. ✅ 整合到 `SynthesizerTrn`
3. ✅ 修改 `TextAudioSpeakerLoader` 支援 emotion prompts
4. ✅ 生成 100 筆測試數據 (自動生成描述)

### Phase 2: 訓練驗證 (2-3 週)

5. ✅ 小規模訓練 (100-1000 筆數據)
6. ✅ 評估情感控制效果
7. ✅ 調整架構 (如需要)

### Phase 3: 規模化 (3-4 週)

8. ✅ 自動生成全部數據的描述
9. ✅ 人工審核高價值樣本 (10-20%)
10. ✅ 完整訓練

---

## 可行性評估

### 技術可行性: ★★★★★ (非常高)

**原因**:
1. CLAP 已經證明了文本-音頻對齊的有效性
2. VITS 架構本身支援多種 conditioning
3. 社群已有類似實作 (PromptTTS, InstructTTS)

### 數據可行性: ★★★☆☆ (中等)

**挑戰**:
- 需要大量情感描述文本
- 自動生成的質量可能不穩定

**解決方案**:
- 先用自動生成打底
- 逐步加入人工審核
- 使用 data augmentation (同一音頻多個描述)

### 計算成本: ★★★★☆ (可接受)

**額外成本**:
- CLAP 推論: ~50ms/sample (可接受)
- 訓練時間增加: ~10-20% (因為多了 emotion encoder)

**優化**:
- 凍結 CLAP 權重
- 使用 smaller CLAP variants
- 預計算 CLAP embeddings (離線)

---

## 與 Reference Audio 方式的對比

| 特性 | Reference Audio | PromptTTS (CLAP) |
|------|----------------|------------------|
| 推論便利性 | ★★☆☆☆ | ★★★★★ |
| 情感控制精度 | ★★★★★ | ★★★★☆ |
| 數據準備難度 | ★★★★★ (容易) | ★★★☆☆ |
| 可解釋性 | ★★☆☆☆ | ★★★★★ |
| 靈活性 | ★★★☆☆ | ★★★★★ |
| 訓練穩定性 | ★★★★★ | ★★★★☆ |

---

## 結論與建議

### 推薦方案

**使用 CLAP-based PromptTTS 風格 + 最小特徵集 (F0 + Energy)**

**理由**:
1. ✅ 符合您的需求 (文本描述而非 reference audio)
2. ✅ 降低特徵維度 (從 97 → 2)
3. ✅ 技術成熟可行
4. ✅ 推論時用戶體驗更好

### 實作優先級

1. **Phase 1** (立即開始):
   - 實作 `MinimalEmotionExtractor` (F0 + Energy)
   - 實作 `CLAPEmotionEncoder`
   - 整合到現有模型

2. **Phase 2** (並行):
   - 自動生成情感描述 (1000 筆測試)
   - 小規模訓練驗證

3. **Phase 3** (後續):
   - 規模化數據準備
   - 完整訓練

### 開放性問題供討論

1. **情感描述的粒度**:
   - 簡單 5 類情感? (happy, sad, angry, neutral, surprised)
   - 還是更細緻的描述? (包含強度、語速等)

2. **多語言支援**:
   - 情感描述用中文還是英文?
   - CLAP 對中文的支援程度?

3. **訓練策略**:
   - 是否先用 reference audio 預訓練,再切換到 prompt?
   - 還是直接端到端訓練?

請告訴我您的想法,我們可以進一步細化設計!
