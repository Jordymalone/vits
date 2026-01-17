#!/usr/bin/env python3
"""
VITS 語音合成推論 CLI
================================================================================
使用方式:
    # 列出可用模型
    python infer.py --list-models

    # 單句合成
    python infer.py --model hakka_hf --text "sil tsh3 iu32 tsh3 in35 sil" --sid 0

    # 批次合成
    python infer.py --model hakka_hf --batch input.txt --output-dir ./output --sid 0

    # 使用預設模型
    python infer.py --text "..." --sid 0
================================================================================
"""

import argparse
import os
import sys

from vits_inferencer import VITSInferencer, list_available_models


def main():
    parser = argparse.ArgumentParser(
        description='VITS 語音合成推論工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 列出可用模型
  python infer.py --list-models

  # 單句合成（使用預設模型）
  python infer.py --text "sil tsh3 iu32 tsh3 in35 sil" --sid 0

  # 指定模型合成
  python infer.py --model hakka_hm --text "an22 ts2 oo22" --sid 0

  # 批次合成
  python infer.py --model hakka_hf --batch input.txt --output-dir ./gen_audio/batch --sid 0
        """
    )
    
    # 模型選項
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='模型名稱（見 --list-models），預設使用 config 中的 default_model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='inference_config.yaml',
        help='配置檔路徑（預設: inference_config.yaml）'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='列出所有可用模型'
    )
    
    # 合成選項
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='要合成的文字（音素序列）'
    )
    parser.add_argument(
        '--sid',
        type=int,
        default=0,
        help='說話人 ID（預設: 0）'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default=None,
        help='語言標籤（ZH/TW/HAK/EN/VI/...），預設使用模型設定'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='輸出檔案路徑（預設: ./gen_audio/{sid}/{output}.wav）'
    )
    
    # 批次選項
    parser.add_argument(
        '--batch', '-b',
        type=str,
        default=None,
        help='批次輸入檔案路徑（格式：檔名|TRN序列）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./gen_audio/batch',
        help='批次輸出目錄（預設: ./gen_audio/batch）'
    )
    parser.add_argument(
        '--no-sil',
        action='store_true',
        help='批次模式不自動加 sil'
    )
    
    # 合成參數
    parser.add_argument(
        '--noise-scale',
        type=float,
        default=0.3,
        help='噪音比例（預設: 0.3）'
    )
    parser.add_argument(
        '--noise-scale-w',
        type=float,
        default=0.3,
        help='噪音權重比例（預設: 0.3）'
    )
    parser.add_argument(
        '--length-scale',
        type=float,
        default=1.4,
        help='長度比例（預設: 1.4），值越大語速越慢'
    )
    
    args = parser.parse_args()
    
    # 處理配置檔路徑
    if not os.path.isabs(args.config):
        args.config = os.path.join(os.path.dirname(__file__), args.config)
    
    # 列出模型
    if args.list_models:
        list_available_models(args.config)
        return
    
    # 檢查必要參數
    if not args.text and not args.batch:
        parser.print_help()
        print("\n錯誤: 請提供 --text 或 --batch 參數")
        sys.exit(1)
    
    # 初始化推論器
    try:
        inferencer = VITSInferencer(
            model_name=args.model,
            config_path=args.config
        )
        inferencer.load_model()
    except Exception as e:
        print(f"錯誤: 無法載入模型 - {e}")
        sys.exit(1)
    
    # 批次模式
    if args.batch:
        inferencer.synthesis_batch(
            input_file=args.batch,
            output_dir=args.output_dir,
            speaker_id=args.sid,
            add_sil=not args.no_sil,
            language=args.lang
        )
        return
    
    # 單句模式
    output_path = args.output
    if not output_path:
        # 預設輸出路徑
        safe_text = args.text[:30].replace(' ', '_').replace('/', '_')
        output_path = f"./gen_audio/{args.sid}/{safe_text}.wav"
    
    try:
        inferencer.synthesis(
            text=args.text,
            speaker_id=args.sid,
            output_path=output_path,
            language=args.lang,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale
        )
    except Exception as e:
        print(f"錯誤: 合成失敗 - {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
