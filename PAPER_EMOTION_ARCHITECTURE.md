# VITS 情緒控制系統架構圖（論文用）

## 系統總覽

本文檔提供完整的 VITS 情緒控制系統架構流程圖，適用於論文撰寫。

---

## 1. 整體系統架構

```mermaid
flowchart TB
    subgraph Input["📥 輸入層"]
        A1[文本序列<br/>Text Sequence]
        A2[說話者 ID<br/>Speaker ID]
        A3[情緒 ID<br/>Emotion ID]
        A4[參考音頻<br/>Reference Audio<br/>可選]
    end

    subgraph Embedding["🔤 嵌入層"]
        B1[文本編碼<br/>Text Encoder]
        B2[說話者嵌入<br/>Speaker Embedding<br/>gin_channels=256]
        B3[情緒嵌入<br/>Emotion Embedding<br/>n_emotions=4]
        B4[eGeMAPS 提取器<br/>eGeMAPS Extractor<br/>可選]
    end

    subgraph Fusion["🔀 特徵融合"]
        C1[向量相加<br/>g = g_speaker + g_emotion]
        C2[eGeMAPS 編碼器<br/>Encoder 88→192 dims<br/>可選]
    end

    subgraph TextEnc["📝 文本編碼器"]
        D1[Transformer Encoder<br/>6 Layers]
        D2[Conditional LayerNorm<br/>CLN]
        D3[Cross Conditional Attention<br/>CCA 可選]
    end

    subgraph Duration["⏱️ 持續時間預測"]
        E1[Stochastic Duration<br/>Predictor SDP]
        E2[Deterministic Duration<br/>Predictor DP]
        E3[混合預測<br/>0.1×SDP + 0.9×DP]
    end

    subgraph Posterior["🎵 後驗編碼器"]
        F1[Posterior Encoder<br/>Mel → Latent z_q]
        F2[條件輸入<br/>g speaker+emotion]
    end

    subgraph Flow["🌊 正規化流"]
        G1[Residual Coupling Blocks<br/>with CLN]
        G2[前向: z_q → z_p<br/>反向: z_p → z_q]
    end

    subgraph Decoder["🔊 解碼器"]
        H1[HiFi-GAN Decoder<br/>with CLN]
        H2[條件輸入<br/>g speaker+emotion]
    end

    subgraph Output["📤 輸出層"]
        I1[合成波形<br/>Waveform]
        I2[注意力對齊<br/>Attention Alignment]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 -.可選.-> B4

    B2 --> C1
    B3 --> C1
    B4 -.可選.-> C2

    B1 --> D1
    C1 --> D2
    C2 -.可選.-> D3
    D1 --> D2
    D2 --> D3
    D3 --> E1
    D3 --> E2

    E1 --> E3
    E2 --> E3

    E3 --> F1
    C1 --> F2
    F2 --> F1

    F1 --> G1
    C1 --> G1
    G1 --> G2

    G2 --> H1
    C1 --> H2
    H2 --> H1

    H1 --> I1
    D3 --> I2

    style Input fill:#e1f5ff
    style Embedding fill:#fff4e1
    style Fusion fill:#ffe1f5
    style TextEnc fill:#e1ffe1
    style Duration fill:#f5e1ff
    style Posterior fill:#ffe1e1
    style Flow fill:#e1e1ff
    style Decoder fill:#ffffe1
    style Output fill:#e1ffff
```

---

## 2. 訓練流程 (Training Pipeline)

```mermaid
flowchart TB
    subgraph Data["📂 數據載入"]
        D1[(Filelist<br/>audio|sid|lang|text|eid)]
        D2[TextAudioSpeakerLoader]
        D3[Batch Collate]
    end

    subgraph Input["📥 訓練輸入"]
        I1[文本 x]
        I2[Mel 頻譜 y]
        I3[說話者 ID sid]
        I4[情緒 ID eid]
    end

    subgraph Forward["⚡ 前向傳播"]
        F1[SynthesizerTrn.forward]
        F2[文本編碼 + CLN/CCA]
        F3[持續時間預測 + CLN]
        F4[Posterior 編碼]
        F5[Flow 正規化]
        F6[HiFi-GAN 解碼]
    end

    subgraph Loss["📉 損失計算"]
        L1[Duration Loss<br/>L_dur]
        L2[Mel Loss<br/>L_mel]
        L3[KL Divergence<br/>L_kl]
        L4[Adversarial Loss<br/>L_adv]
        L5[Feature Matching<br/>L_fm]
        L6[總損失<br/>L_total]
    end

    subgraph Optimize["🔄 優化"]
        O1[Generator 優化器]
        O2[Discriminator 優化器]
        O3[更新參數]
    end

    D1 --> D2
    D2 --> D3
    D3 --> I1 & I2 & I3 & I4

    I1 & I2 & I3 & I4 --> F1
    F1 --> F2 --> F3 --> F4 --> F5 --> F6

    F3 --> L1
    F6 --> L2
    F5 --> L3
    F6 --> L4 & L5

    L1 & L2 & L3 & L4 & L5 --> L6

    L6 --> O1 & O2
    O1 & O2 --> O3
    O3 -.迭代.-> F1

    style Data fill:#e1f5ff
    style Input fill:#fff4e1
    style Forward fill:#e1ffe1
    style Loss fill:#ffe1e1
    style Optimize fill:#f5e1ff
```

---

## 3. 推論流程 (Inference Pipeline)

```mermaid
flowchart TB
    subgraph UserInput["👤 用戶輸入"]
        U1[文本<br/>Text]
        U2[說話者 ID<br/>Speaker ID]
        U3[情緒 ID<br/>Emotion ID]
        U4[參考音頻<br/>Reference Audio<br/>可選]
    end

    subgraph Preprocess["🔧 預處理"]
        P1[文本 → 音素序列]
        P2[插入空白符]
        P3[轉換為 Tensor]
    end

    subgraph Embed["🔤 嵌入"]
        E1[說話者嵌入<br/>emb_g sid]
        E2[情緒嵌入<br/>emb_e eid]
        E3[向量融合<br/>g = g_s + g_e]
        E4[eGeMAPS 提取<br/>可選]
    end

    subgraph Encode["📝 編碼"]
        EN1[Text Encoder<br/>with CLN]
        EN2[CCA 注入<br/>可選]
        EN3[輸出均值 m_p<br/>和方差 logs_p]
    end

    subgraph Predict["⏱️ 預測持續時間"]
        PR1[SDP + DP 混合]
        PR2[生成對齊路徑<br/>Monotonic Alignment]
        PR3[上採樣到音頻幀]
    end

    subgraph Sample["🎲 採樣"]
        S1[從 N m_p, logs_p<br/>採樣 z_p]
        S2[Flow 反向<br/>z_p → z]
    end

    subgraph Decode["🔊 解碼"]
        DE1[HiFi-GAN Decoder<br/>with g condition]
        DE2[生成波形]
    end

    subgraph Output["📤 輸出"]
        O1[合成語音<br/>Audio Waveform]
        O2[注意力圖<br/>Attention Map]
    end

    U1 --> P1
    P1 --> P2
    P2 --> P3

    U2 --> E1
    U3 --> E2
    U4 -.可選.-> E4
    E1 & E2 --> E3

    P3 & E3 --> EN1
    E4 -.可選.-> EN2
    EN1 --> EN2
    EN2 --> EN3

    EN3 --> PR1
    E3 --> PR1
    PR1 --> PR2
    PR2 --> PR3

    EN3 --> S1
    PR3 --> S1
    S1 --> S2
    E3 --> S2

    S2 --> DE1
    E3 --> DE1
    DE1 --> DE2

    DE2 --> O1
    PR2 --> O2

    style UserInput fill:#e1f5ff
    style Preprocess fill:#fff4e1
    style Embed fill:#ffe1f5
    style Encode fill:#e1ffe1
    style Predict fill:#f5e1ff
    style Sample fill:#e1e1ff
    style Decode fill:#ffffe1
    style Output fill:#e1ffff
```

---

## 4. 情緒控制機制詳解

```mermaid
flowchart TB
    subgraph EmotionInput["🎭 情緒輸入"]
        EI1[離散情緒 ID<br/>eid ∈ 0,1,2,3]
        EI2[連續聲學特徵<br/>eGeMAPS 可選]
    end

    subgraph EmotionEmbed["🔢 情緒表示"]
        EE1[Emotion Embedding<br/>eid → R^256]
        EE2[eGeMAPS Encoder<br/>R^88 → R^192]
    end

    subgraph Condition["🎯 條件注入"]
        C1[Conditional LayerNorm<br/>CLN]
        C2[Cross Conditional<br/>Attention CCA]
    end

    subgraph CLN_Detail["📐 CLN 機制"]
        CL1[標準化<br/>x_norm = x - μ / σ]
        CL2[條件調製<br/>γ_c, β_c = f g]
        CL3[輸出<br/>y = x_norm × 1+γ_c + β_c]
    end

    subgraph CCA_Detail["🔗 CCA 機制"]
        CA1[Query 來自文本<br/>Q = W_q × x_text]
        CA2[Key, Value 來自情緒<br/>K,V = W_k,v × emo_feat]
        CA3[注意力計算<br/>Attn Q,K × V]
        CA4[殘差連接<br/>x + Attn]
    end

    subgraph Impact["💫 情緒影響"]
        IM1[持續時間<br/>Duration]
        IM2[音高韻律<br/>Pitch/F0]
        IM3[能量響度<br/>Energy]
        IM4[頻譜特徵<br/>Spectral]
    end

    EI1 --> EE1
    EI2 -.可選.-> EE2

    EE1 --> C1
    EE2 -.可選.-> C2

    C1 --> CL1
    CL1 --> CL2
    CL2 --> CL3

    C2 --> CA1
    C2 --> CA2
    CA2 --> CA3
    CA1 --> CA3
    CA3 --> CA4

    CL3 --> IM1 & IM2 & IM3 & IM4
    CA4 --> IM1 & IM2 & IM3 & IM4

    style EmotionInput fill:#ffe1f5
    style EmotionEmbed fill:#fff4e1
    style Condition fill:#e1ffe1
    style CLN_Detail fill:#e1f5ff
    style CCA_Detail fill:#f5e1ff
    style Impact fill:#ffffe1
```

---

## 5. 條件層歸一化 (CLN) 詳細流程

```mermaid
flowchart LR
    subgraph Input["輸入"]
        I1[特徵 x<br/>B,C,T]
        I2[條件 g<br/>B,gin_ch,1]
    end

    subgraph Norm["標準化"]
        N1[計算均值 μ]
        N2[計算方差 σ²]
        N3[x_norm =<br/>x - μ / √ σ² + ε]
    end

    subgraph Condition["條件生成"]
        C1[Conv1D<br/>gin_ch → 2×C]
        C2[分割為<br/>γ_c 和 β_c]
    end

    subgraph Modulate["條件調製"]
        M1[縮放<br/>x_norm × 1 + γ_c]
        M2[平移<br/>+ β_c]
    end

    subgraph Output["輸出"]
        O1[調製後特徵<br/>y B,C,T]
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
    style Condition fill:#ffe1f5
    style Modulate fill:#e1ffe1
    style Output fill:#ffffe1
```

**數學公式**:

$$
\begin{align}
\text{CLN}(x, g) &= \gamma_c \odot \text{LN}(x) + \beta_c \\
[\gamma_c, \beta_c] &= \text{Conv1D}(g) \\
\text{LN}(x) &= \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
\end{align}
$$

---

## 6. 交叉條件注意力 (CCA) 詳細流程

```mermaid
flowchart TB
    subgraph Input["📥 輸入"]
        I1[文本特徵<br/>x_text B,C,T_text]
        I2[情緒特徵<br/>emo_feat B,C_emo,T_emo]
    end

    subgraph Projection["🔀 投影"]
        P1[Query 投影<br/>Q = Conv_q x_text<br/>B,C,T_text]
        P2[Key 投影<br/>K = Conv_k emo_feat<br/>B,C,T_emo]
        P3[Value 投影<br/>V = Conv_v emo_feat<br/>B,C,T_emo]
    end

    subgraph MultiHead["🔢 多頭分割"]
        M1[Q → Q_1,...,Q_h<br/>h=4 heads]
        M2[K → K_1,...,K_h]
        M3[V → V_1,...,V_h]
    end

    subgraph Attention["⚡ 注意力計算"]
        A1[對每個頭 i:<br/>Attn_i = softmax Q_i K_i^T / √d_k]
        A2[Out_i = Attn_i × V_i]
        A3[拼接所有頭<br/>Out = Concat Out_1,...,Out_h]
    end

    subgraph Output["📤 輸出"]
        O1[輸出投影<br/>y = Conv_o Out]
        O2[殘差連接<br/>x_text + y]
        O3[LayerNorm<br/>LN x_text + y]
    end

    I1 --> P1
    I2 --> P2 & P3

    P1 --> M1
    P2 --> M2
    P3 --> M3

    M1 & M2 --> A1
    A1 --> A2
    M3 --> A2
    A2 --> A3

    A3 --> O1
    O1 --> O2
    I1 --> O2
    O2 --> O3

    style Input fill:#e1f5ff
    style Projection fill:#fff4e1
    style MultiHead fill:#ffe1f5
    style Attention fill:#e1ffe1
    style Output fill:#ffffe1
```

**數學公式**:

$$
\begin{align}
\text{CCA}(x, c) &= \text{LN}(x + \text{MultiHead}(Q, K, V)) \\
Q &= W_q x, \quad K = W_k c, \quad V = W_v c \\
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1, ..., \text{head}_h)W_o \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{align}
$$

---

## 7. 持續時間預測機制

```mermaid
flowchart TB
    subgraph Input["📥 輸入"]
        I1[文本編碼<br/>x B,C,T]
        I2[條件向量<br/>g B,gin_ch,1]
    end

    subgraph SDP["🎲 隨機預測器 (SDP)"]
        S1[Transformer Encoder<br/>with CLN]
        S2[Projection<br/>C → 1]
        S3[Flow Matching<br/>學習分布]
        S4[輸出 log_dur_sdp]
    end

    subgraph DP["📏 確定性預測器 (DP)"]
        D1[Conv Blocks<br/>with CLN]
        D2[ReLU + Dropout]
        D3[Projection<br/>C → 1]
        D4[輸出 log_dur_dp]
    end

    subgraph Combine["🔀 混合策略"]
        C1[訓練時:<br/>log_dur = log_dur_gt<br/>用於對齊]
        C2[推論時:<br/>log_dur = 0.1×sdp + 0.9×dp]
        C3[exp log_dur<br/>獲得持續時間]
    end

    subgraph Alignment["🎯 對齊生成"]
        A1[Monotonic Alignment<br/>Search MAS]
        A2[生成對齊路徑<br/>attn T_text,T_audio]
        A3[上採樣文本特徵<br/>x_expanded]
    end

    I1 & I2 --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4

    I1 & I2 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4

    S4 --> C1
    D4 --> C1
    C1 --> C2
    C2 --> C3

    C3 --> A1
    I1 --> A1
    A1 --> A2
    A2 --> A3

    style Input fill:#e1f5ff
    style SDP fill:#ffe1f5
    style DP fill:#fff4e1
    style Combine fill:#e1ffe1
    style Alignment fill:#ffffe1
```

**情緒對持續時間的影響**:
- **Happy**: 持續時間 ↓ (語速快)
- **Sad**: 持續時間 ↑ (語速慢)
- **Angry**: 持續時間 ↓ (語速快、急促)
- **Neutral**: 基準持續時間

---

## 8. 數據流程

```mermaid
flowchart TB
    subgraph Raw["📂 原始數據"]
        R1[音頻檔案<br/>*.wav]
        R2[文本標註<br/>*.txt]
        R3[情緒標籤<br/>metadata]
    end

    subgraph Prepare["🔧 數據準備"]
        P1[音素化<br/>G2P]
        P2[情緒映射<br/>Label → ID]
        P3[生成 Filelist<br/>prepare_emotion_filelist.py]
    end

    subgraph Filelist["📋 Filelist"]
        F1[格式:<br/>path|sid|lang|phonemes|eid]
        F2[訓練集<br/>emotion_train.txt]
        F3[驗證集<br/>emotion_val.txt]
    end

    subgraph Loader["📥 數據載入"]
        L1[TextAudioSpeakerLoader]
        L2[讀取音頻]
        L3[計算 Mel 頻譜]
        L4[文本編碼]
    end

    subgraph Batch["📦 Batch 處理"]
        B1[BucketSampler<br/>相似長度分組]
        B2[TextAudioSpeakerCollate<br/>填充對齊]
        B3[輸出 Batch<br/>x,spec,wav,sid,eid]
    end

    subgraph Training["🎓 訓練"]
        T1[送入模型<br/>SynthesizerTrn]
    end

    R1 & R2 & R3 --> P1
    P1 --> P2
    P2 --> P3

    P3 --> F1
    F1 --> F2 & F3

    F2 & F3 --> L1
    L1 --> L2 & L3 & L4

    L2 & L3 & L4 --> B1
    B1 --> B2
    B2 --> B3

    B3 --> T1

    style Raw fill:#e1f5ff
    style Prepare fill:#fff4e1
    style Filelist fill:#ffe1f5
    style Loader fill:#e1ffe1
    style Batch fill:#f5e1ff
    style Training fill:#ffffe1
```

---

## 9. 模型組件層次結構

```mermaid
flowchart TB
    subgraph Model["🎯 SynthesizerTrn"]
        direction TB

        subgraph Encoders["編碼器"]
            E1[TextEncoder<br/>enc_p]
            E2[PosteriorEncoder<br/>enc_q]
        end

        subgraph Predictors["預測器"]
            P1[DurationPredictor<br/>dp]
            P2[StochasticDP<br/>sdp]
        end

        subgraph Transform["轉換器"]
            T1[ResidualCouplingBlock<br/>flow]
        end

        subgraph Generator["生成器"]
            G1[HiFi-GAN Decoder<br/>dec]
        end

        subgraph Embeddings["嵌入層"]
            EM1[Speaker Embedding<br/>emb_g]
            EM2[Emotion Embedding<br/>emb_e]
        end

        subgraph Optional["可選模組"]
            O1[eGeMAPS Extractor<br/>egemaps_extractor]
            O2[eGeMAPS Encoder<br/>egemaps_encoder]
        end
    end

    Encoders -.使用.-> Embeddings
    Predictors -.使用.-> Embeddings
    Transform -.使用.-> Embeddings
    Generator -.使用.-> Embeddings

    Encoders -.可選使用.-> Optional

    style Model fill:#e1f5ff
    style Encoders fill:#fff4e1
    style Predictors fill:#ffe1f5
    style Transform fill:#e1ffe1
    style Generator fill:#f5e1ff
    style Embeddings fill:#ffffe1
    style Optional fill:#e1e1ff
```

---

## 10. 情緒特徵提取 (eGeMAPS - 可選)

```mermaid
flowchart TB
    subgraph Input["📥 參考音頻"]
        I1[波形<br/>Waveform B,T_wav]
    end

    subgraph Extract["🔊 特徵提取"]
        E1[MFCC<br/>13 維]
        E2[Mel-spectrogram<br/>80 維]
        E3[F0 基頻<br/>1 維]
        E4[Energy 能量<br/>1 維]
        E5[Spectral Flux<br/>1 維]
        E6[Zero Crossing Rate<br/>1 維]
    end

    subgraph Concat["🔗 特徵拼接"]
        C1[總特徵向量<br/>97 維]
        C2[MLP 投影<br/>97 → 88 維]
    end

    subgraph Encode["📊 編碼"]
        EN1[Conv1D Encoder<br/>3 層]
        EN2[LayerNorm + ReLU]
        EN3[輸出<br/>B,192,T_feat]
    end

    subgraph Usage["🎯 使用方式"]
        U1[輸入到 CCA<br/>作為條件特徵]
        U2[與文本特徵<br/>交叉注意力]
    end

    I1 --> E1 & E2 & E3 & E4 & E5 & E6
    E1 & E2 & E3 & E4 & E5 & E6 --> C1
    C1 --> C2

    C2 --> EN1
    EN1 --> EN2
    EN2 --> EN3

    EN3 --> U1
    U1 --> U2

    style Input fill:#e1f5ff
    style Extract fill:#fff4e1
    style Concat fill:#ffe1f5
    style Encode fill:#e1ffe1
    style Usage fill:#ffffe1
```

**注意**: 本實作使用純 Label ID，eGeMAPS 為可選增強功能。

---

## 11. 損失函數架構

```mermaid
flowchart TB
    subgraph Losses["💰 損失函數"]
        direction TB

        subgraph Generator["🎨 生成器損失"]
            G1[Duration Loss<br/>L_dur = MSE dur_pred, dur_gt]
            G2[Mel Loss<br/>L_mel = L1 mel_pred, mel_gt]
            G3[KL Divergence<br/>L_kl = KL z_p || z_q]
            G4[Adversarial Loss<br/>L_adv_g = -E log D y_fake]
            G5[Feature Matching<br/>L_fm = Σ ||f_real - f_fake||]
        end

        subgraph Discriminator["🔍 判別器損失"]
            D1[Real Loss<br/>L_real = -E log D y_real]
            D2[Fake Loss<br/>L_fake = -E log 1-D y_fake]
            D3[Total D Loss<br/>L_d = L_real + L_fake]
        end

        subgraph Total["📊 總損失"]
            T1[Generator Total<br/>L_g = L_dur + c_mel×L_mel +<br/>c_kl×L_kl + L_adv + L_fm]
            T2[權重<br/>c_mel=45, c_kl=1.0]
        end
    end

    G1 & G2 & G3 & G4 & G5 --> T1
    T2 -.配置.-> T1

    style Losses fill:#e1f5ff
    style Generator fill:#e1ffe1
    style Discriminator fill:#ffe1e1
    style Total fill:#ffffe1
```

**損失函數數學表達**:

$$
\begin{align}
\mathcal{L}_{\text{dur}} &= \text{MSE}(\log d_{\text{pred}}, \log d_{\text{gt}}) \\
\mathcal{L}_{\text{mel}} &= ||M_{\text{pred}} - M_{\text{gt}}||_1 \\
\mathcal{L}_{\text{kl}} &= \text{KL}(q(z|x) || p(z)) \\
\mathcal{L}_{\text{adv}} &= -\mathbb{E}[\log D(G(z))] \\
\mathcal{L}_{\text{fm}} &= \sum_{i=1}^{T} \frac{1}{N_i}||D^i(y) - D^i(\hat{y})||_1 \\
\mathcal{L}_{\text{G}} &= \mathcal{L}_{\text{dur}} + c_{\text{mel}}\mathcal{L}_{\text{mel}} + c_{\text{kl}}\mathcal{L}_{\text{kl}} + \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{fm}}
\end{align}
$$

---

## 12. 評估與測試流程

```mermaid
flowchart TB
    subgraph Test["🧪 測試設置"]
        T1[載入模型<br/>Checkpoint]
        T2[設定測試文本<br/>Test Text]
        T3[設定參數<br/>speaker, emotions]
    end

    subgraph Generate["🎵 生成多種情緒"]
        G1[Neutral ID=0]
        G2[Happy ID=1]
        G3[Sad ID=2]
        G4[Angry ID=3]
    end

    subgraph Analyze["📊 客觀分析"]
        A1[持續時間<br/>Duration]
        A2[能量<br/>RMS Energy]
        A3[基頻<br/>F0 Estimation]
        A4[最大振幅<br/>Max Amplitude]
    end

    subgraph Compare["📈 對比分析"]
        C1[計算相對差異<br/>vs Neutral %]
        C2[生成統計表格]
        C3[輸出 JSON 報告]
    end

    subgraph Subjective["👂 主觀評估"]
        S1[聽覺測試<br/>Listening Test]
        S2[情緒可辨識度<br/>Emotion Recognition]
        S3[自然度評分<br/>Naturalness MOS]
    end

    T1 & T2 & T3 --> G1 & G2 & G3 & G4

    G1 & G2 & G3 & G4 --> A1 & A2 & A3 & A4

    A1 & A2 & A3 & A4 --> C1
    C1 --> C2
    C2 --> C3

    G1 & G2 & G3 & G4 --> S1
    S1 --> S2 & S3

    style Test fill:#e1f5ff
    style Generate fill:#fff4e1
    style Analyze fill:#e1ffe1
    style Compare fill:#ffe1f5
    style Subjective fill:#ffffe1
```

---

## 13. 完整系統資訊流

```mermaid
flowchart LR
    subgraph Stage1["階段 1: 輸入"]
        S1A[文本]
        S1B[說話者]
        S1C[情緒]
    end

    subgraph Stage2["階段 2: 嵌入"]
        S2A[文本編碼]
        S2B[說話者向量]
        S2C[情緒向量]
    end

    subgraph Stage3["階段 3: 條件融合"]
        S3A[g = g_s + g_e]
        S3B[CLN 調製]
        S3C[CCA 注入]
    end

    subgraph Stage4["階段 4: 編碼"]
        S4A[文本 → 隱向量]
        S4B[持續時間預測]
        S4C[對齊生成]
    end

    subgraph Stage5["階段 5: 生成"]
        S5A[Flow 採樣]
        S5B[HiFi-GAN 解碼]
    end

    subgraph Stage6["階段 6: 輸出"]
        S6A[語音波形]
    end

    S1A --> S2A
    S1B --> S2B
    S1C --> S2C

    S2B & S2C --> S3A
    S3A --> S3B & S3C

    S2A & S3B & S3C --> S4A
    S4A --> S4B
    S4B --> S4C

    S4C & S3A --> S5A
    S5A & S3A --> S5B

    S5B --> S6A

    style Stage1 fill:#e1f5ff
    style Stage2 fill:#fff4e1
    style Stage3 fill:#ffe1f5
    style Stage4 fill:#e1ffe1
    style Stage5 fill:#f5e1ff
    style Stage6 fill:#ffffe1
```

---

## 14. 代碼架構對應

### 核心文件映射

```mermaid
flowchart TB
    subgraph Code["💻 代碼架構"]
        direction LR

        subgraph Core["核心模型"]
            C1[models.py<br/>SynthesizerTrn]
            C2[modules.py<br/>CLN, Layers]
            C3[attentions.py<br/>CCA, Attention]
        end

        subgraph Emotion["情緒組件"]
            E1[Emotion Embedding<br/>models.py:509-510]
            E2[eGeMAPS Extractor<br/>egemaps_extractor.py]
            E3[eGeMAPS Encoder<br/>egemaps_extractor.py]
        end

        subgraph Data["數據處理"]
            D1[data_utils.py<br/>Loader + Collate]
            D2[text/<br/>Text Processing]
            D3[mel_processing.py<br/>Mel Computation]
        end

        subgraph Train["訓練與推論"]
            T1[train_ms.py<br/>Training Loop]
            T2[infer.py<br/>Inference]
            T3[losses.py<br/>Loss Functions]
        end

        subgraph Utils["輔助工具"]
            U1[prepare_emotion_filelist.py<br/>Data Preparation]
            U2[test_emotion_control.py<br/>Testing]
        end
    end

    C1 -.uses.-> C2 & C3
    C1 -.contains.-> E1
    C1 -.optional.-> E2 & E3
    T1 -.uses.-> C1 & D1
    T2 -.uses.-> C1 & D2

    style Code fill:#e1f5ff
    style Core fill:#e1ffe1
    style Emotion fill:#ffe1f5
    style Data fill:#fff4e1
    style Train fill:#f5e1ff
    style Utils fill:#ffffe1
```

### 關鍵類別與方法

| 組件 | 文件位置 | 關鍵方法 |
|------|---------|---------|
| **SynthesizerTrn** | models.py:415-825 | `forward()`, `infer()` |
| **TextEncoder** | models.py:133-196 | `forward()` with CCA |
| **DurationPredictor** | models.py:80-130 | `forward()` with CLN |
| **ConditionalLayerNorm** | modules.py:34-75 | `forward()` |
| **CrossConditionalAttention** | attentions.py:257-345 | `forward()` |
| **eGeMAPS_Extractor** | egemaps_extractor.py:20-150 | `extract_*()`, `forward()` |
| **TextAudioSpeakerLoader** | data_utils.py:160-287 | `__getitem__()`, `_filter()` |

---

## 15. 論文圖表建議

### 建議的論文圖表順序

1. **系統總覽圖** (圖 1)
   - 使用「整體系統架構」流程圖
   - 展示完整的輸入到輸出流程

2. **情緒控制機制** (圖 2)
   - 使用「情緒控制機制詳解」
   - 重點標註 CLN 和 CCA

3. **CLN 架構** (圖 3)
   - 使用「條件層歸一化詳細流程」
   - 配合數學公式

4. **CCA 架構** (圖 4)
   - 使用「交叉條件注意力詳細流程」
   - 展示多頭注意力機制

5. **訓練流程** (圖 5)
   - 使用「訓練流程」
   - 標註損失函數

6. **推論流程** (圖 6)
   - 使用「推論流程」
   - 展示用戶如何控制情緒

7. **實驗結果** (表格)
   - 客觀指標對比表
   - 主觀評估 MOS 表

### LaTeX 圖表引用範例

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{emotion_control_architecture.pdf}
\caption{VITS 情緒控制系統整體架構。系統接收文本、說話者 ID 和情緒 ID 作為輸入，通過 Conditional LayerNorm (CLN) 和 Cross Conditional Attention (CCA) 機制將情緒信息注入到模型各層，最終生成具有指定情緒的語音波形。}
\label{fig:system_overview}
\end{figure}
```

---

## 16. 關鍵技術貢獻

```mermaid
mindmap
  root((VITS<br/>情緒控制))
    貢獻 1
      Emotion Embedding
        離散情緒 ID
        n_emotions=4
        向量維度 256
    貢獻 2
      Conditional LayerNorm
        動態調製 γ, β
        應用於所有層
        情緒條件注入
    貢獻 3
      Cross Conditional Attention
        多頭注意力
        文本-情緒交互
        殘差連接
    貢獻 4
      Duration Prediction
        SDP + DP 混合
        CLN 條件化
        情緒影響韻律
    貢獻 5
      端到端訓練
        單階段訓練
        聯合優化
        無需預訓練
```

---

## 參考文獻建議

### 相關工作

1. **VITS**: Kim et al., "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech", ICML 2021

2. **Conditional LayerNorm**: Dumoulin et al., "A Learned Representation for Artistic Style", ICLR 2017

3. **Emotion TTS**:
   - Skerry-Ryan et al., "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron", ICML 2018
   - Valle et al., "Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis", ICLR 2021

4. **eGeMAPS**: Eyben et al., "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing", IEEE Trans. Affective Computing, 2016

---

## 附錄：符號表

| 符號 | 說明 | 維度 |
|------|------|------|
| $x$ | 文本序列 | $[B, T_{text}]$ |
| $y$ | 音頻波形 | $[B, T_{audio}]$ |
| $\text{sid}$ | 說話者 ID | $[B]$ |
| $\text{eid}$ | 情緒 ID | $[B]$ |
| $g$ | 全局條件向量 | $[B, 256, 1]$ |
| $z$ | 隱變量 | $[B, 192, T]$ |
| $m_p, \log s_p$ | 先驗均值、方差 | $[B, 192, T]$ |
| $m_q, \log s_q$ | 後驗均值、方差 | $[B, 192, T]$ |
| $d$ | 持續時間 | $[B, T_{text}]$ |
| $C$ | 隱藏維度 | 192 |
| $C_{gin}$ | 條件維度 | 256 |

---

**本架構圖使用 Mermaid 語法生成，可在支援 Mermaid 的 Markdown 編輯器中渲染，或使用工具轉換為 PDF/PNG 格式用於論文。**
