"""
VITS Inferencer Module
封裝 VITS 模型載入和推論邏輯
"""

import os
import time
import yaml
import torch
import numpy as np
import soundfile as sf

import commons
import utils
from models import SynthesizerTrn


class VITSInferencer:
    """VITS 語音合成推論器"""
    
    def __init__(self, model_name: str = None, config_path: str = "inference_config.yaml"):
        """
        初始化推論器
        
        Args:
            model_name: 模型名稱（對應 config 中的 key），若為 None 則使用 default_model
            config_path: 配置檔路徑
        """
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name or self.config.get('default_model')
        self.model_config = self.config['models'].get(self.model_name)
        
        if not self.model_config:
            available = list(self.config['models'].keys())
            raise ValueError(f"Model '{self.model_name}' not found. Available: {available}")
        
        self.net_g = None
        self.hps = None
        self.symbols = None
        self.symbol_to_id = {}  # 動態建立的 symbol mapping
        self.speaker_dict = {}
        
    def load_model(self):
        """載入模型和相關資源"""
        # 取得相對於腳本的路徑
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        config_path = os.path.join(script_dir, self.model_config['config'])
        checkpoint_path = os.path.join(script_dir, self.model_config['checkpoint'])
        lang_phones_path = os.path.join(script_dir, self.model_config['lang_phones'])
        speaker_file_path = os.path.join(script_dir, self.model_config['speaker_file'])
        
        # 載入 hyperparameters
        self.hps = utils.get_hparams_from_file(config_path)
        
        # 建立 symbols
        self.symbols = self._build_symbols(lang_phones_path)
        
        # 建立模型
        self.net_g = SynthesizerTrn(
            len(self.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        ).cuda()
        
        _ = self.net_g.eval()
        
        # 載入 checkpoint
        _ = utils.load_checkpoint(checkpoint_path, self.net_g, None)
        
        # 載入 speaker dict
        self._load_speaker_dict(speaker_file_path)
        
        print(f"[VITSInferencer] 模型已載入: {self.model_config.get('name', self.model_name)}")
        print(f"[VITSInferencer] Symbols 數量: {len(self.symbols)}")
        
        return self
    
    def _build_symbols(self, lang_phones_path: str) -> list:
        """建立 symbol 列表和 symbol_to_id mapping"""
        _pad = '_'
        _punctuation = ';:,.!?¡¿—…-–\"«»"" '
        _tone = '0123456789'
        _sym = []
        
        with open(lang_phones_path, 'r', encoding='utf-8') as f:
            phonemes = [p.strip() for p in f.readlines()]
        
        symbols = list(phonemes) + [_pad] + list(_punctuation) + list(_tone)
        symbols += _sym * 2
        
        # 建立 symbol_to_id mapping
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        
        return symbols
    
    def _load_speaker_dict(self, speaker_file_path: str):
        """載入 speaker ID 對照表"""
        if not os.path.exists(speaker_file_path):
            return
            
        with open(speaker_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    self.speaker_dict[parts[1]] = parts[0]
    
    def _cleaned_text_to_sequence(self, cleaned_text: str, language: str) -> list:
        """
        將音素文字轉換為 ID 序列（使用動態載入的 symbol mapping）
        
        Args:
            cleaned_text: 清理過的音素序列
            language: 語言標籤
        """
        sequence = []
        cleaned_text = cleaned_text.strip().replace("  ", " ")
        
        # 使用 phoneme 語言時，按空格分割後查表
        for ph in cleaned_text.split(' '):
            ph = ph.strip()
            if not ph:
                continue
            if ph not in self.symbol_to_id:
                raise ValueError(
                    f'Symbol not found! Original phoneme string: "{cleaned_text}", '
                    f'problematic segment: "{ph}" not found in symbol_to_id map.'
                )
            sequence.append(self.symbol_to_id[ph])
        
        return sequence
    
    def _get_text(self, text: str, language: str) -> torch.LongTensor:
        """將文字轉換為模型輸入張量"""
        text_norm = self._cleaned_text_to_sequence(text, language)
        text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    
    def _process_text(self, text: str) -> str:
        """預處理文字"""
        return text.lower()
    
    def synthesis(
        self,
        text: str,
        speaker_id: int,
        output_path: str = None,
        language: str = None,
        noise_scale: float = 0.3,
        noise_scale_w: float = 0.3,
        length_scale: float = 1.4
    ) -> np.ndarray:
        """
        合成語音
        
        Args:
            text: 輸入文字（音素序列）
            speaker_id: 說話人 ID
            output_path: 輸出檔案路徑（可選）
            language: 語言標籤，若為 None 則使用模型預設
            noise_scale: 噪音比例
            noise_scale_w: 噪音權重比例
            length_scale: 長度比例
            
        Returns:
            合成的音訊 numpy array
        """
        if self.net_g is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        language = language or self.model_config.get('default_language', 'TW')
        processed_text = self._process_text(text)
        processed_text = processed_text.replace(", ", ",")
        
        result_segments = []
        start_time = time.time()
        
        for segment in processed_text.split(","):
            segment = segment.strip()
            if not segment:
                continue
            
            stn_tst = self._get_text(segment, language)
            
            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                sid = torch.LongTensor([int(speaker_id)]).cuda()
                
                o, attn, _, _ = self.net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale
                )
                
                audio = o[0, 0].data.cpu().float().numpy()
                result_segments.append(audio)
        
        if not result_segments:
            raise ValueError("No valid text segments to synthesize")
        
        concatenated_audio = np.concatenate(result_segments)
        elapsed_time = time.time() - start_time
        
        print(f"[VITSInferencer] 合成完成，耗時 {elapsed_time:.2f} 秒")
        
        # 儲存音訊
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            sf.write(output_path, concatenated_audio, 22050)
            print(f"[VITSInferencer] 已儲存至: {output_path}")
        
        return concatenated_audio
    
    def synthesis_batch(
        self,
        input_file: str,
        output_dir: str,
        speaker_id: int,
        add_sil: bool = True,
        language: str = None
    ):
        """
        批次合成語音
        
        Args:
            input_file: 輸入檔案路徑（格式：檔名|TRN序列）
            output_dir: 輸出目錄
            speaker_id: 說話人 ID
            add_sil: 是否在前後加 sil
            language: 語言標籤
        """
        print(f"{'='*70}")
        print(f"批次語音合成")
        print(f"{'='*70}")
        print(f"模型: {self.model_config.get('name', self.model_name)}")
        print(f"輸入檔案: {input_file}")
        print(f"輸出目錄: {output_dir}")
        print(f"說話人 ID: {speaker_id}")
        print(f"加 sil: {add_sil}")
        print(f"{'='*70}\n")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入檔案: {input_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        error_log_path = os.path.join(output_dir, "error_list.txt")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total = len(lines)
        success_count = 0
        fail_count = 0
        errors = []
        
        for idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split("|", 1)
                if len(parts) != 2:
                    raise ValueError(f"格式錯誤: {line}")
                
                file_id, text = parts[0].strip(), parts[1].strip()
                
                if add_sil:
                    if not text.startswith('sil'):
                        text = 'sil ' + text
                    if not text.endswith('sil'):
                        text = text + ' sil'
                
                output_path = os.path.join(output_dir, f"{file_id}.wav")
                
                print(f"[{idx:3d}/{total}] 合成中: {file_id}.wav...", end=' ')
                
                self.synthesis(
                    text=text,
                    speaker_id=speaker_id,
                    output_path=output_path,
                    language=language
                )
                
                print("✓")
                success_count += 1
                
            except Exception as e:
                print(f"✗ 錯誤: {e}")
                errors.append(f"Line {idx}: {line}\nError: {str(e)}\n")
                fail_count += 1
        
        # 寫入錯誤記錄
        if errors:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.writelines(errors)
        
        print(f"\n{'='*70}")
        print(f"合成完成！")
        print(f"{'='*70}")
        print(f"✓ 成功: {success_count} 個 ({success_count/total*100:.1f}%)")
        print(f"✗ 失敗: {fail_count} 個 ({fail_count/total*100:.1f}%)")
        if fail_count > 0:
            print(f"錯誤記錄: {error_log_path}")
        print(f"{'='*70}")
    
    @classmethod
    def list_models(cls, config_path: str = "inference_config.yaml") -> dict:
        """列出所有可用模型"""
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('models', {})


def list_available_models(config_path: str = "inference_config.yaml"):
    """列出並印出所有可用模型"""
    models = VITSInferencer.list_models(config_path)
    
    print(f"\n{'='*70}")
    print("可用模型列表")
    print(f"{'='*70}\n")
    
    # 按語言分組
    groups = {}
    for key, model in models.items():
        lang = model.get('default_language', 'OTHER')
        if lang not in groups:
            groups[lang] = []
        groups[lang].append((key, model))
    
    lang_names = {
        'HAK': '客家語',
        'TW': '台語',
        'ZH': '國語/中文',
        'EN': '英語',
        'VI': '越南語',
        'OTHER': '其他'
    }
    
    for lang, items in groups.items():
        print(f"【{lang_names.get(lang, lang)}】")
        for key, model in items:
            name = model.get('name', key)
            print(f"  - {key}: {name}")
        print()
    
    print(f"{'='*70}")
    print(f"共 {len(models)} 個模型")
    print(f"{'='*70}\n")
