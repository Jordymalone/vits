# Emotion-Controllable VITS - Flowchart 架構圖

> 📅 生成日期：2026-01-09  
> 🎯 用途：論文架構圖 (Mermaid Flowchart)

---

## 1. 完整系統架構 (System Overview)

```mermaid
flowchart TB
    subgraph Input["📥 輸入層 Input Layer"]
        I1[Text/Phoneme<br/>文字序列]
        I2[Speaker ID<br/>說話者]
        I3[Emotion ID<br/>情緒標籤]
        I4[Reference Audio<br/>參考音頻 可選]
    end

    subgraph Embedding["🔤 嵌入層 Embedding Layer"]
        E1[Speaker Embedding<br/>emb_g: 256-dim]
        E2[Emotion Embedding<br/>emb_e: 256-dim]
        E3["g = g_spk + g_emo<br/>向量融合"]
    end

    subgraph eGeMAPS["🎵 eGeMAPS Pipeline 可選"]
        G1[eGeMAPS Extractor<br/>F0/Energy/MFCC/Mel]
        G2[Feature Projection<br/>97 → 88 dims]
        G3[eGeMAPS Encoder<br/>88 → 192 dims]
    end

    subgraph TextEnc["📝 Text Encoder"]
        T1[Phoneme Embedding]
        T2[Transformer × 6<br/>+ CLN]
        T3[CCA Module<br/>可選]
        T4["Prior: m_p, logs_p"]
    end

    subgraph Duration["⏱️ Duration Predictor"]
        D1[SDP: Stochastic]
        D2[DP: Deterministic]
        D3["Mix: 0.1×SDP + 0.9×DP"]
    end

    subgraph Flow["🌊 Normalizing Flow"]
        F1[Residual Coupling × 4]
    end

    subgraph Decoder["🔊 HiFi-GAN Decoder"]
        DEC1[Upsampling × 4]
        DEC2[MRF ResBlocks]
        DEC3[Waveform Output]
    end

    I1 --> T1
    I2 --> E1
    I3 --> E2
    I4 -.-> G1

    E1 --> E3
    E2 --> E3

    G1 --> G2 --> G3

    T1 --> T2
    E3 --> T2
    T2 --> T3
    G3 -.-> T3
    T3 --> T4

    T4 --> D1
    T4 --> D2
    E3 --> D1
    E3 --> D2
    D1 --> D3
    D2 --> D3

    D3 --> F1
    E3 --> F1

    F1 --> DEC1
    E3 --> DEC1
    DEC1 --> DEC2
    DEC2 --> DEC3

    style Input fill:#e1f5ff
    style Embedding fill:#fff4e1
    style eGeMAPS fill:#ffe1f5
    style TextEnc fill:#e1ffe1
    style Duration fill:#f5e1ff
    style Flow fill:#e1e1ff
    style Decoder fill:#ffffe1
```

---

## 2. 訓練流程 (Training Pipeline)

```mermaid
flowchart TB
    subgraph DataLoad["📂 數據載入"]
        DL1[(Filelist<br/>audio|sid|lang|text|eid)]
        DL2[TextAudioSpeakerLoader]
        DL3[Batch Collate]
    end

    subgraph Forward["⚡ 前向傳播"]
        FW1[Text Encoder<br/>+ CLN + CCA]
        FW2[Posterior Encoder<br/>Mel → z_q]
        FW3[Flow: z_q → z_p]
        FW4[MAS Alignment]
        FW5[Duration Prediction]
        FW6[HiFi-GAN Decode]
    end

    subgraph Loss["📉 損失計算"]
        L1["L_mel = ||Mel_real - Mel_fake||₁"]
        L2["L_kl = KL(z_q || z_p)"]
        L3["L_dur = MSE(log_dur)"]
        L4["L_adv = GAN Loss"]
        L5["L_fm = Feature Matching"]
        L6["L_total = L_mel×45 + L_kl + L_dur + L_adv + L_fm"]
    end

    subgraph Optim["🔄 優化"]
        O1[Generator Optimizer<br/>AdamW lr=2e-4]
        O2[Discriminator Optimizer<br/>AdamW lr=2e-4]
    end

    DL1 --> DL2 --> DL3
    DL3 --> FW1
    FW1 --> FW4
    FW2 --> FW3
    FW3 --> FW4
    FW4 --> FW5
    FW5 --> FW6

    FW6 --> L1
    FW3 --> L2
    FW5 --> L3
    FW6 --> L4
    FW6 --> L5
    L1 & L2 & L3 & L4 & L5 --> L6

    L6 --> O1
    L4 --> O2
    O1 & O2 -.->|迭代| FW1

    style DataLoad fill:#e1f5ff
    style Forward fill:#e1ffe1
    style Loss fill:#ffe1e1
    style Optim fill:#f5e1ff
```

---

## 3. 推論流程 (Inference Pipeline)

```mermaid
flowchart TB
    subgraph Input["👤 用戶輸入"]
        IN1[文本 Text]
        IN2[Speaker ID]
        IN3[Emotion ID<br/>或 Reference Audio]
    end

    subgraph Step1["Step 1: 嵌入"]
        S1A["g_spk = Embedding(sid)"]
        S1B["g_emo = Embedding(eid)"]
        S1C["g = g_spk + g_emo"]
    end

    subgraph Step2["Step 2: eGeMAPS 可選"]
        S2A[eGeMAPS Extract]
        S2B[eGeMAPS Encode]
    end

    subgraph Step3["Step 3: 文本編碼"]
        S3A[Text Encoder<br/>with CLN]
        S3B[CCA Attention<br/>可選]
        S3C["Output: m_p, logs_p"]
    end

    subgraph Step4["Step 4: 時長預測"]
        S4A["log_dur = 0.1×SDP + 0.9×DP"]
        S4B["dur = ceil(exp(log_dur) × length_scale)"]
    end

    subgraph Step5["Step 5: 對齊擴展"]
        S5A[Generate Path]
        S5B["Expand m_p, logs_p to T_audio"]
    end

    subgraph Step6["Step 6: 採樣"]
        S6A["z_p = m_p + ε × exp(logs_p) × noise_scale"]
    end

    subgraph Step7["Step 7: Flow 反向"]
        S7A["z = Flow⁻¹(z_p, g)"]
    end

    subgraph Step8["Step 8: 解碼"]
        S8A[HiFi-GAN Decoder]
        S8B[🔊 Waveform Output]
    end

    IN1 --> S3A
    IN2 --> S1A
    IN3 --> S1B
    IN3 -.-> S2A

    S1A --> S1C
    S1B --> S1C

    S2A --> S2B
    S2B -.-> S3B

    S1C --> S3A
    S3A --> S3B
    S3B --> S3C

    S3C --> S4A
    S1C --> S4A
    S4A --> S4B

    S4B --> S5A
    S3C --> S5A
    S5A --> S5B

    S5B --> S6A

    S6A --> S7A
    S1C --> S7A

    S7A --> S8A
    S1C --> S8A
    S8A --> S8B

    style Input fill:#e1f5ff
    style Step1 fill:#fff4e1
    style Step2 fill:#ffe1f5
    style Step3 fill:#e1ffe1
    style Step4 fill:#f5e1ff
    style Step5 fill:#e1e1ff
    style Step6 fill:#ffffe1
    style Step7 fill:#e1ffff
    style Step8 fill:#ffe1e1
```

---

## 4. Conditional Layer Normalization (CLN) 機制

```mermaid
flowchart LR
    subgraph Input["輸入"]
        I1["x: 特徵<br/>[B, C, T]"]
        I2["g: 條件<br/>[B, 256, 1]"]
    end

    subgraph Norm["LayerNorm"]
        N1["μ = mean(x)"]
        N2["σ² = var(x)"]
        N3["x_norm = (x - μ) / √(σ² + ε)"]
    end

    subgraph Cond["條件生成"]
        C1["Conv1d(g)<br/>256 → 2×C"]
        C2["Split → γ_c, β_c"]
    end

    subgraph Modulate["調製"]
        M1["Scale: x_norm × (1 + γ_c)"]
        M2["Shift: + β_c"]
    end

    subgraph Output["輸出"]
        O1["y: 調製後特徵<br/>[B, C, T]"]
    end

    I1 --> N1 & N2
    N1 & N2 --> N3

    I2 --> C1
    C1 --> C2

    N3 --> M1
    C2 --> M1
    M1 --> M2
    C2 --> M2

    M2 --> O1

    style Input fill:#e1f5ff
    style Norm fill:#fff4e1
    style Cond fill:#ffe1f5
    style Modulate fill:#e1ffe1
    style Output fill:#ffffe1
```

**數學公式：**
$$\text{CLN}(x, g) = (1 + \gamma_c) \odot \text{LN}(x) + \beta_c$$
$$[\gamma_c, \beta_c] = \text{Conv1D}(g)$$

---

## 5. Cross-Conditional Attention (CCA) 機制

```mermaid
flowchart TB
    subgraph Input["輸入"]
        I1["x_text: 文本特徵<br/>[B, 192, T_text]"]
        I2["emo_feat: 情緒特徵<br/>[B, 192, T_emo]"]
    end

    subgraph Projection["投影層"]
        P1["Q = Conv_q(x_text)<br/>Query from Text"]
        P2["K = Conv_k(emo_feat)<br/>Key from Emotion"]
        P3["V = Conv_v(emo_feat)<br/>Value from Emotion"]
    end

    subgraph Attention["Multi-Head Attention"]
        A1["Reshape to<br/>[B, n_heads, T, d_k]"]
        A2["scores = Q × K^T / √d_k"]
        A3["attn = Softmax(scores)"]
        A4["out = attn × V"]
    end

    subgraph Output["輸出"]
        O1["Conv_o: 輸出投影"]
        O2["Residual: x_text + out"]
        O3["LayerNorm"]
    end

    I1 --> P1
    I2 --> P2 & P3

    P1 --> A1
    P2 --> A1
    P3 --> A1

    A1 --> A2
    A2 --> A3
    A3 --> A4

    A4 --> O1
    O1 --> O2
    I1 --> O2
    O2 --> O3

    style Input fill:#e1f5ff
    style Projection fill:#fff4e1
    style Attention fill:#ffe1f5
    style Output fill:#e1ffe1
```

**數學公式：**
$$\text{CCA}(x, c) = \text{LN}(x + \text{MultiHead}(Q, K, V))$$
$$Q = W_q \cdot x, \quad K = W_k \cdot c, \quad V = W_v \cdot c$$

---

## 6. eGeMAPS 特徵提取流程

```mermaid
flowchart TB
    subgraph Input["📥 輸入"]
        I1["Reference Audio<br/>[B, T_wav]"]
    end

    subgraph Extract["🎵 特徵提取"]
        E1["Mel-Spectrogram<br/>80 dims"]
        E2["MFCC<br/>13 dims"]
        E3["F0 基頻<br/>1 dim"]
        E4["Energy 能量<br/>1 dim"]
        E5["Spectral Flux<br/>1 dim"]
        E6["Zero Crossing Rate<br/>1 dim"]
    end

    subgraph Concat["🔗 拼接"]
        C1["Total: 97 dims<br/>[B, 97, T_frames]"]
    end

    subgraph Project["📊 投影"]
        P1["MLP: 97 → 256 → 88<br/>[B, 88, T_frames]"]
    end

    subgraph Encode["🔧 編碼"]
        EN1["Pre Conv1d: 88 → 192"]
        EN2["Conv Block × 3<br/>+ LayerNorm + ReLU<br/>+ Residual"]
        EN3["Output: 192 dims<br/>[B, 192, T_frames]"]
    end

    subgraph Usage["🎯 使用"]
        U1["→ CCA Module<br/>作為 Key/Value"]
    end

    I1 --> E1 & E2 & E3 & E4 & E5 & E6
    E1 & E2 & E3 & E4 & E5 & E6 --> C1
    C1 --> P1
    P1 --> EN1
    EN1 --> EN2
    EN2 --> EN3
    EN3 --> U1

    style Input fill:#e1f5ff
    style Extract fill:#fff4e1
    style Concat fill:#ffe1f5
    style Project fill:#e1ffe1
    style Encode fill:#f5e1ff
    style Usage fill:#ffffe1
```

---

## 7. Duration Prediction 機制

```mermaid
flowchart TB
    subgraph Input["📥 輸入"]
        I1["x: Text Encoding<br/>[B, 192, T_text]"]
        I2["g: Global Condition<br/>[B, 256, 1]"]
    end

    subgraph SDP["🎲 Stochastic Duration Predictor"]
        S1["Transformer Layer<br/>+ CLN"]
        S2["DDSConv Processing"]
        S3["Flow Coupling × 4"]
        S4["log_dur_sdp"]
    end

    subgraph DP["📏 Deterministic Duration Predictor"]
        D1["Conv Layer 1<br/>+ CLN + ReLU"]
        D2["Conv Layer 2<br/>+ CLN + ReLU"]
        D3["Projection → 1"]
        D4["log_dur_dp"]
    end

    subgraph Mix["🔀 混合策略"]
        M1["log_dur = 0.1 × SDP + 0.9 × DP"]
        M2["dur = ceil(exp(log_dur) × length_scale)"]
    end

    subgraph Align["🎯 對齊"]
        A1["Generate Monotonic Path"]
        A2["Expand Features to T_audio"]
    end

    I1 --> S1
    I2 --> S1
    S1 --> S2 --> S3 --> S4

    I1 --> D1
    I2 --> D1
    D1 --> D2 --> D3 --> D4

    S4 --> M1
    D4 --> M1
    M1 --> M2
    M2 --> A1
    A1 --> A2

    style Input fill:#e1f5ff
    style SDP fill:#ffe1f5
    style DP fill:#fff4e1
    style Mix fill:#e1ffe1
    style Align fill:#ffffe1
```

---

## 8. 情緒控制模式對比

```mermaid
flowchart TB
    subgraph ModeA["模式 A: Emotion Embedding"]
        A1["Emotion ID<br/>(0=neutral, 1=happy, 2=sad, 3=angry)"]
        A2["Embedding Table<br/>[n_emotions, 256]"]
        A3["g_emo → CLN"]
        A1 --> A2 --> A3
    end

    subgraph ModeB["模式 B: eGeMAPS + CCA"]
        B1["Reference Audio"]
        B2["eGeMAPS Features<br/>[B, 88, T]"]
        B3["eGeMAPS Encoder<br/>[B, 192, T]"]
        B4["CCA with Text"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph ModeC["模式 C: 混合模式"]
        C1["Emotion ID → CLN<br/>粗粒度控制"]
        C2["Reference Audio → CCA<br/>細粒度控制"]
        C3["雙重情緒注入"]
        C1 --> C3
        C2 --> C3
    end

    subgraph Effect["💫 效果"]
        E1["Duration 變化<br/>Happy: 快 / Sad: 慢"]
        E2["F0 Pitch 變化<br/>Happy: 高 / Sad: 低"]
        E3["Energy 變化<br/>Angry: 高 / Sad: 低"]
    end

    ModeA --> Effect
    ModeB --> Effect
    ModeC --> Effect

    style ModeA fill:#e1f5ff
    style ModeB fill:#ffe1f5
    style ModeC fill:#e1ffe1
    style Effect fill:#ffffe1
```

---

## 9. 整體架構圖 (簡化版)

```mermaid
flowchart LR
    subgraph In["Input"]
        Text
        SpeakerID
        EmotionID
        RefAudio
    end

    subgraph Enc["Encoding"]
        TextEnc["Text Encoder<br/>+ CLN + CCA"]
        EmoEnc["eGeMAPS Encoder"]
    end

    subgraph Pred["Prediction"]
        DurPred["Duration<br/>Predictor"]
        Flow["Normalizing<br/>Flow"]
    end

    subgraph Dec["Decoding"]
        HiFiGAN["HiFi-GAN<br/>Decoder"]
    end

    subgraph Out["Output"]
        Waveform["🔊 Audio"]
    end

    Text --> TextEnc
    SpeakerID --> TextEnc
    EmotionID --> TextEnc
    RefAudio -.-> EmoEnc
    EmoEnc -.-> TextEnc

    TextEnc --> DurPred
    TextEnc --> Flow
    DurPred --> Flow

    Flow --> HiFiGAN
    HiFiGAN --> Waveform

    style In fill:#e1f5ff
    style Enc fill:#e1ffe1
    style Pred fill:#fff4e1
    style Dec fill:#ffe1f5
    style Out fill:#ffffe1
```

---

## 10. 論文用架構圖 (Paper-Ready)

```mermaid
flowchart TB
    subgraph Training["Training Phase"]
        direction TB
        T_Text["Text x"] --> T_TextEnc["Text Encoder<br/>+ CLN"]
        T_Mel["Mel y"] --> T_PosEnc["Posterior<br/>Encoder"]
        T_Spk["Speaker ID"] --> T_Emb["Embeddings"]
        T_Emo["Emotion ID"] --> T_Emb
        T_Ref["Ref Audio"] -.-> T_eGeMAPS["eGeMAPS"]
        
        T_eGeMAPS -.-> T_CCA["CCA"]
        T_TextEnc --> T_CCA
        T_CCA --> T_Prior["Prior<br/>m_p, logs_p"]
        
        T_PosEnc --> T_Flow["Flow"]
        T_Prior --> T_MAS["MAS"]
        T_Flow --> T_MAS
        
        T_MAS --> T_DurPred["Duration<br/>Predictor"]
        T_Flow --> T_Dec["Decoder"]
        T_Dec --> T_Out["ŷ"]
        
        T_Emb --> T_TextEnc
        T_Emb --> T_PosEnc
        T_Emb --> T_Flow
        T_Emb --> T_Dec
    end

    subgraph Inference["Inference Phase"]
        direction TB
        I_Text["Text x"] --> I_TextEnc["Text Encoder<br/>+ CLN"]
        I_Spk["Speaker ID"] --> I_Emb["Embeddings"]
        I_Emo["Emotion/Ref"] --> I_Emb
        
        I_Emb --> I_TextEnc
        I_TextEnc --> I_DurPred["Duration<br/>Predictor"]
        I_DurPred --> I_Expand["Expand"]
        I_TextEnc --> I_Expand
        I_Expand --> I_Sample["Sample z_p"]
        I_Sample --> I_Flow["Flow⁻¹"]
        I_Flow --> I_Dec["Decoder"]
        I_Dec --> I_Out["ŷ"]
        
        I_Emb --> I_Flow
        I_Emb --> I_Dec
    end

    style Training fill:#e1f5ff
    style Inference fill:#e1ffe1
```

---

*Generated: 2026-01-09*
