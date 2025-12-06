這是一份針對此專案的技術架構導覽

#### 1\. 核心模型層 (Model Core)

這些檔案定義了神經網絡的具體結構。

| 檔案名稱 | 關鍵類別/功能 | 說明 |
| :--- | :--- | :--- |
| **`models.py`** | `SynthesizerTrn` | **VITS 主體**。整合了 TextEncoder, PosteriorEncoder, Flow, Decoder。 |
| | `Generator` | **HiFi-GAN Decoder**。負責將潛在變數 $z$ 轉回波形。 |
| | `TextEncoder` | 處理文本輸入，包含 Transformer based 的注意力機制。 |
| | `PosteriorEncoder` | (僅訓練用) 處理頻譜圖輸入，基於 WaveNet 架構。 |
| | `StochasticDurationPredictor` | 隨機時長預測器，讓語音節奏具多樣性。 |
| **`modules.py`** | `ResBlock1`, `WN` | 定義基礎卷積塊、WaveNet 層、殘差塊。 |
| **`attentions.py`** | `MultiHeadAttention` | Transformer 的多頭注意力機制實作。 |
| **`commons.py`** | `sequence_mask` | 通用工具，如處理變長序列的 Mask、切片操作等。 |
| **`losses.py`** | `generator_loss`, `kl_loss` | 定義 VITS 的損失函數：GAN Loss, Feature Match Loss, Mel Loss, KL Divergence, Duration Loss。 |

#### 2\. 數據與預處理層 (Data & Preprocessing)

負責將原始音訊與文本轉換為模型可吃的 Tensor。

| 檔案名稱 | 功能 | 說明 |
| :--- | :--- | :--- |
| **`data_utils.py`** | `TextAudioSpeakerLoader` | PyTorch Dataset 實作。讀取音訊路徑、文本 ID，並進行 Batch 處理。 |
| | `DistributedBucketSampler` | 針對不同長度的音訊進行分桶（Bucket），減少 Padding 浪費，加速訓練。 |
| **`mel_processing.py`** | `mel_spectrogram_torch` | 將波形轉為 Mel-Spectrogram (梅爾頻譜圖)。 |
| **`preprocess.py`** | (通常是 main) | 雖有名為 preprocess，但 VITS 通常是 On-the-fly 處理。此腳本可能用於預先計算頻譜或清洗數據。 |
| **`text/` (資料夾)** | **文本前端** | 整個 TTS 的入口。 |
| ├ `symbols.py` | `symbols` | **極重要**。定義所有可用的音素列表。模型輸入維度依賴於此。 |
| ├ `cleaners.py` | `english_cleaners2` | 文本正規化（如將數字轉英文、縮寫展開）。 |
| └ `__init__.py` | `text_to_sequence` | 將字串轉為 ID 序列。 |

#### 3\. 訓練與執行層 (Execution)

控制訓練迴圈與推論邏輯。

| 檔案名稱 | 功能 | 說明 |
| :--- | :--- | :--- |
| **`train.py`** | 單機/單語者訓練 | 適用於 LJSpeech 等單語者數據集。 |
| **`train_ms.py`** | **多語者訓練** | 適用於 VCTK 等多語者數據集。包含 Speaker Embedding 的處理邏輯。 |
| **`infer.py`** | 推論腳本 | 載入 Checkpoint 並合成語音的範例。 |
| **`inference.ipynb`** | Jupyter Notebook | 互動式推論範例，通常用於快速測試。 |

#### 4\. 對齊機制 (Alignment)

VITS 的黑科技：Monotonic Alignment Search (MAS)。

| 檔案名稱 | 功能 | 說明 |
| :--- | :--- | :--- |
| **`monotonic_align/`** | **MAS 算法** | 用於在沒有人工標註時間點的情況下，自動學習文本與音訊的對齊路徑。 |
| ├ `core.pyx` | Cython 程式碼 | 為了效能，核心動態規劃算法是用 Cython 寫的，需編譯。 |
| └ `setup.py` | 編譯腳本 | 用於編譯 `.pyx` 檔。 |

#### 5\. 設定檔 (Configuration)

控制模型超參數。

| 檔案名稱 | 內容 | 說明 |
| :--- | :--- | :--- |
| **`configs/`** | `.json` 檔案 | 包含 `model` (層數、維度), `data` (路徑、採樣率), `train` (Batch size, LR) 等設定。 |

-----

### 總結

這個專案結構非常標準。若要深入閱讀代碼，建議順序如下：

1.  **`configs/*.json`**：先看有哪些參數。
2.  **`models.py` (`SynthesizerTrn.__init__`)**：看模型如何讀取這些參數建立架構。
3.  **`models.py` (`SynthesizerTrn.forward`)**：看訓練時資料如何流動（包含 PosteriorEncoder）。
4.  **`models.py` (`SynthesizerTrn.infer`)**：看推論時如何避開 PosteriorEncoder，僅用 Prior 生成。