import argparse
import torch
import commons
import utils
from models import SynthesizerTrn
# from text.symbols_re import symbols
from text.symbols import symbols
from text import text_to_sequence, cleaned_text_to_sequence
import numpy as np
from scipy.io.wavfile import write
import time
import re
import os
import librosa
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch
import soundfile as sf
from tools.phonemes_transformation.hakka import hakka_segment
# 建立兩個客家話前端：四縣腔(ID=2)與海陸腔(ID=3)
sixsian_frontend = hakka_segment.hakka_frontend(language_id=2)
hailu_frontend   = hakka_segment.hakka_frontend(language_id=3)
import json

speaker_dict = {}

def load_speaker_dict():
    # with open(f'/home/p76111652/Linux_DATA/synthesis/corpus/22050/dataset/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/double_phoneme_zh_tw/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/3646_vad_25_920/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/Hakka_hm/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/hakka_wo_hac/mixed_5_id.txt', 'r') as f:
    with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/Hakka_xm/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/hakka_six_segment_4096/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/Hakka_hf/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/hakka_six_v1/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/phonetic_test_zh/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/tw_1115_doublevVowel/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/tw_1220_kaldi_300_noise/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/retraintw/mixed_5_id.txt', 'r') as f:

        for line in f.readlines():
            line = line.strip().split("|")
            speaker_dict[line[1]] = line[0]

def process_text(input_text, english_flag):
    '''
    原本某次訓練越南語以小寫輸入為主，後來英文沒注意也是以小寫，目前兩個模型是分開的，故我先取消這邊的更動，都轉成小寫
    '''
    # print("input_text: ", input_text)
    input_text = input_text.lower()
    if not english_flag:
        input_text = input_text.lower()
    # output_text = re.sub(r'[^\w\s]', ',', input_text)
    return input_text

def get_text(text, langauge):
    cleaner_names = ['zh_cleaners']
    text_norm = cleaned_text_to_sequence(text, langauge)

    text_norm = commons.intersperse(text_norm, 0)

    text_norm = torch.LongTensor(text_norm)
    return text_norm

# hps = utils.get_hparams_from_file("logs/tw_new_2/config.json")
# hps = utils.get_hparams_from_file("logs/3834_5_vad/config.json")
# hps = utils.get_hparams_from_file("logs/3646_vad_25_920/config.json")
# hps = utils.get_hparams_from_file("logs/hakka_six_v1/config.json")
# hps = utils.get_hparams_from_file("logs/Hakka_single_xf/config.json")
hps = utils.get_hparams_from_file("logs/Hakka_single_xm/config.json")
# hps = utils.get_hparams_from_file("logs/hakka_hm_model/config.json")
# hps = utils.get_hparams_from_file("logs/hakka_hf_model/config.json")
# hps = utils.get_hparams_from_file("logs/hakka_wo_hac/config.json")
# hps = utils.get_hparams_from_file("logs/tw_1220_kaldi_300_noise/config.json")
# hps = utils.get_hparams_from_file("logs/hakka_six_segment_4096/config.json")
# hps = utils.get_hparams_from_file("logs/phonetic_test_zh/config.json")


def get_symbols():
    _pad        = '_'
    _punctuation = ';:,.!?¡¿—…-–"«»“” '
    _tone = '0123456789'
    # 240816 竟烽要留的
    # _sym = [' ', '!', ',', '.', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̩', 'θ', 'ᵻ']
    # 竟烽訓練chracter synthesis用的
    # _sym = [' ', '', "'", ',', '.', '1', '2', '3', '4', '5', '6', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ô', 'ă', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɣ', 'ɤ', 'ɪ', 'ɯ', 'ɲ', 'ʂ', 'ʃ', 'ʈ', 'ʊ', 'ʐ', 'ʧ', 'ʰ', 'ʷ', 'ˈ', 'ˌ', '̆', '͡', 'ầ']
    # 241104 景霈訓練單台語用的
    # _sym = ['', ',', '0', '1', '2', '3', '4', '5', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'y', 'z']
    # 241106 景霈訓練雙母音+nycu語者台語
    # 因為lang_chars.txt是空的

    # 3646_vad_25_920
    # _sym = [
    #     ' ', '!', '"', ',', '.', ':', ';', '?',
    #     'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm',
    #     'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z',
    #     'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ',
    #     'ɡ', 'ɪ', 'ɬ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʲ',
    #     'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', 'ᵻ'
    # ]
    _sym = []
    # 250109 景霈訓練純英語
    # _sym = ['', '!', '"', ',', '.', ':', ';', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʲ', 'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', 'ᵻ']

    # with open(f'filelists/3834_5_retrainpaintako_vad/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/3646_vad_25_920/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/hakka_six_v1/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/Hakka_hm/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/Hakka_hf/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/hakka_wo_hac/lang_phones.txt','r',encoding='utf-8') as f:
    with open(f'filelists/Hakka_xm/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/tw_1220_kaldi_300_noise/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/hakka_six_segment_4096/lang_phones.txt','r',encoding='utf-8') as f:
    # with open(f'filelists/phonetic_test_zh/lang_phones.txt','r',encoding='utf-8') as f:

        phonemes = f.readlines()
        phonemes = [p.strip() for p in phonemes]

    # Export all symbols:
    # symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_tone) + list(_extra_indo) + list(_special)
    symbols = list(phonemes) + [_pad] + list(_punctuation) + list(_tone)
    base_offset = len(symbols)
    sym_offset = len(_sym)
    symbols += _sym * 2
    # symbols += _sym * 1
    SPACE_ID = symbols.index(" ")
    return symbols

symbols = get_symbols()
print("here is len symbols len", len(symbols))
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

# 新tw 用 421000
# 新en 用 607000
# use_g_pth_file = 'G_696000' # retraintw
# use_g_pth_file = 'G_159000' # Hakka_single_xf
use_g_pth_file = 'G_305000' # Hakka_single_xm
# use_g_pth_file = 'G_519000' # Hakka_hf
# use_g_pth_file = 'G_350000' # retrain 3834_5_vad
# use_g_pth_file = 'G_253000' # Hakka_hm
# use_g_pth_file = 'G_300000' # hakka_wo_hac
# use_g_pth_file = 'G_200000' # finetune_tw_1220_kaldi_300_noise
# use_g_pth_file = 'G_120000' # hakka_six_segment_4096
# use_g_pth_file = 'G_344000' # hakka_six_v1
# use_g_pth_file = 'G_480000' # 3646_vad_25_920
# use_g_pth_file = 'G_253000' # hakka_hm_model
# use_g_pth_file = 'G_90000' # phonetic_test_zh
# use_g_pth_file = 'G_192000' # 0512_vad_test
# use_g_pth_file = 'G_286000' # tw_1220_kaldi_300_noise
# use_g_pth_file = 'G_290000' # 1115雙母音+nycu語者台語
# _ = utils.load_checkpoint("logs/tw_new_2/G_568000.pth", net_g, None) # 應該是之前測試別的專案寫死的
# _ = utils.load_checkpoint(f"logs/phonetic_test_zh/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/3834_5_vad/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/3646_vad_25_920/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/hakka_six_segment_4096/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/hakka_wo_hac/{use_g_pth_file}.pth", net_g, None)
_ = utils.load_checkpoint(f"logs/Hakka_single_xm/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/finetune_tw_1220_kaldi_300_noise/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/tw_1220_kaldi_300_noise/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/hakka_hm_model/{use_g_pth_file}.pth", net_g, None)
# _ = utils.load_checkpoint(f"logs/hakka_hf_model/{use_g_pth_file}.pth", net_g, None)


def save_attn_txt(attn_tensor, save_path):
    attn = attn_tensor[0, 0].cpu().numpy()  # [T_audio, T_text]
    np.savetxt(save_path, attn, fmt="%.4f", delimiter=",")

def synthesis(text, speaker_id, speaker_name, filename, english_flag, langauge):
    processed_text = process_text(text, english_flag)
    result_np_arr = []
    start_time = time.time()
    processed_text = processed_text.replace(", ",",")
    attn_index = 0

    for each in str(processed_text).split(","):
        # prevent None
        if not each:
            continue
        each = each.strip()
        stn_tst = get_text(each, langauge)
        print("get_text output tensor:", stn_tst)
        # print("length:", stn_tst.size(0))

        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([int(speaker_id)]).cuda()
            # # audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.0, noise_scale_w=0.1, length_scale=1.15)[0][0,0].data.cpu().float().numpy()
            # o, attn, _, _ = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.3, noise_scale_w=0.3, length_scale=1.4)
            # # o, attn, _, _ = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.3, noise_scale_w=0.3, length_scale=1.2)    # 0513 vad test
            # audio = o[0, 0].data.cpu().float().numpy()
            # result_np_arr.append(audio)
            o, attn, _, _ = net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.3,
                noise_scale_w=0.3,
                length_scale=1.4
            )
            audio = o[0, 0].data.cpu().float().numpy()
            result_np_arr.append(audio)

            # === DEBUG: 存 attn 成 CSV（選擇性） ===
            # attn: [B, 1, T_y, T_x]
            debug_dir = "./attn_debug"
            os.makedirs(debug_dir, exist_ok=True)
            attn_np = attn[0, 0].data.cpu().numpy()  # [T_y, T_x]
            attn_csv_path = os.path.join(debug_dir, f"{speaker_id}_{filename}_attn.csv")
            np.savetxt(attn_csv_path, attn_np, fmt="%.4f", delimiter=",")
            print(f"[DEBUG] attn saved to {attn_csv_path}")

            # 儲存對齊文字檔
            # attn_txt_path = f"./gen_audio/{speaker_name}/{filename}_attn_{attn_index}.csv"
            # save_attn_txt(attn, attn_txt_path)
            # attn_index += 1

    # print("test attn shape: ", attn.shape)
    # print("test attn: ", attn)
    concatenated_audio = np.concatenate(result_np_arr)
    elapsed_time = time.time() - start_time
    print(f"Synthesis for speaker {speaker_id} took {elapsed_time} seconds.")
    if not os.path.exists(f'./gen_audio'):
        os.makedirs(f'./gen_audio')
    if not os.path.exists(f'./gen_audio/{speaker_name}'):
        os.makedirs(f'./gen_audio/{speaker_name}')

    audio_path = f"./gen_audio/{speaker_name}/{filename}"
    os.makedirs(os.path.dirname(audio_path), exist_ok = True)
    sf.write(audio_path, concatenated_audio, 22050)
    return audio_path

def synthesis_file_with_sil(input_file, output_dir, speaker_id, add_sil=True):
    """
    批次合成語音，支援在每句前後加 sil
    
    Args:
        input_file: 輸入檔案路徑（格式：檔名|TRN序列）
        output_dir: 輸出目錄
        speaker_id: 說話人 ID
        add_sil: 是否在前後加 sil（預設 True）
    """
    print(f"{'='*70}")
    print(f"批次語音合成（本地推論）")
    print(f"{'='*70}")
    print(f"輸入檔案: {input_file}")
    print(f"輸出目錄: {output_dir}")
    print(f"說話人 ID: {speaker_id}")
    print(f"加 sil: {add_sil}")
    print(f"{'='*70}\n")
    
    # 確認輸入檔案存在
    if not os.path.exists(input_file):
        print(f"✗ 錯誤：找不到輸入檔案 {input_file}")
        return
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 準備錯誤記錄檔
    error_log_path = os.path.join(output_dir, "error_list.txt")
    error_list = open(error_log_path, "w", encoding="utf8")
    
    # 讀取輸入檔案
    with open(input_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    total = len(lines)
    success_count = 0
    fail_count = 0
    
    print(f"開始合成 {total} 個音檔...")
    print(f"{'-'*70}\n")
    
    for idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            # 解析格式：檔名|TRN序列
            parts = line.split("|", 1)
            if len(parts) != 2:
                error_msg = f"格式錯誤: {line}"
                print(f"✗ [{idx:3d}/{total}] {error_msg}")
                error_list.write(f"Line {idx}: {error_msg}\n")
                fail_count += 1
                continue
            
            file_id, text = parts
            file_id = file_id.strip()
            text = text.strip()
            
            # 在前後加 sil
            if add_sil:
                if not text.startswith('sil'):
                    text = 'sil ' + text
                if not text.endswith('sil'):
                    text = text + ' sil'
            
            # 輸出檔名
            output_filename = f"{file_id}.wav"
            
            print(f"[{idx:3d}/{total}] 正在合成: {output_filename}...", end=' ')
            
            # 呼叫 synthesis（使用 HAK 語言標籤）
            audio_path = synthesis(
                text=text,
                speaker_id=speaker_id,
                speaker_name=output_dir.split('/')[-1],  # 使用目錄名作為 speaker_name
                filename=output_filename,
                english_flag=False,
                langauge='HAK'
            )
            
            # 移動檔案到指定目錄
            generated_path = audio_path
            target_path = os.path.join(output_dir, output_filename)
            
            if os.path.exists(generated_path) and generated_path != target_path:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                os.rename(generated_path, target_path)
            
            print(f"✓")
            success_count += 1
            
            # 每 10 個顯示進度
            if idx % 10 == 0:
                print(f"--- 進度: {idx}/{total} ({idx/total*100:.1f}%) ---")
            
        except Exception as e:
            error_msg = f"{line}\nError: {str(e)}"
            print(f"✗ [{idx:3d}/{total}] 錯誤: {e}")
            error_list.write(f"Line {idx}: {error_msg}\n\n")
            fail_count += 1
            continue
    
    error_list.close()
    
    # 顯示統計結果
    print(f"\n{'='*70}")
    print(f"合成完成！")
    print(f"{'='*70}")
    print(f"✓ 成功: {success_count} 個 ({success_count/total*100:.1f}%)")
    print(f"✗ 失敗: {fail_count} 個 ({fail_count/total*100:.1f}%)")
    if fail_count > 0:
        print(f"錯誤記錄已儲存至: {error_log_path}")
    print(f"{'='*70}")

def get_audio(filename):
    audio, _ = load_wav_to_torch(filename)
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")
    spec = spectrogram_torch(audio_norm, 1024, 22050, 256, 1024, center=False)
    spec = torch.squeeze(spec, 0)
    torch.save(spec, spec_filename)
    return spec, audio_norm

def vc(reference_path: os.path, model):
    spec, audio_norm = get_audio(reference_path)
    spec = spec.unsqueeze(0).cuda()
    spec_lengths = torch.LongTensor([spec.size(2)]).cuda().float()
    sid_src = torch.LongTensor([56]).cuda() # 56 是交大 speaker
    target_speakerId = 31 # 王育德

    with torch.no_grad():
        sid_tgt1 = torch.LongTensor([target_speakerId]).cuda()
        print(f'Processing speaker {target_speakerId}')
        audio1 = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data.cpu().float().numpy()
        print(f'105: {audio1.shape}')
        write_path = reference_path.split("/")[-1].replace(".wav", f"_{target_speakerId}.wav")
        print(f'Writing to {write_path}')
        sf.write(write_path, audio1, 22050, 'PCM_16')


# def synthesis_file(language, speaker_id, net_g):
#     with open(f'gen_text/{language}.txt', 'r', encoding='utf8') as f:
#         lines = f.readlines()

#     language_dict = {
#         'zh': 'ZH',
#         'ctl': 'TZH',
#         'ha': 'HAK',
#         'tw': 'TW',
#         'en': 'EN',
#         'id': 'ID',
#         'jp' : 'JP',
#         'vi' : 'VI'
#     }

#     for line in lines:
#         filename, text = line.strip().split("|")
#         audio_path = synthesis(text, speaker_id, speaker_id, filename, False, language_dict[language])
#         # vc(audio_path, net_g)

language_dict = {
    'zh' : 'ZH',
    'ctl': 'TZH',
    'ha' : 'HAK',
    'tw' : 'TW',
    'en' : 'EN',
    'id' : 'ID',
    'jp' : 'JP',
    'vi' : 'VI'
}

def synthesis_file(language, speaker_id, net_g):
    lang_tag = language_dict[language]

    # 準備錯誤記錄檔
    error_list = open("error_list.txt", "a", encoding="utf8")

    with open(f'gen_text/{language}.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        try:
            filename, text = line.strip().split("|")

            # 呼叫 synthesis
            audio_path = synthesis(
                text,
                speaker_id,
                speaker_id,
                filename,
                False,
                lang_tag
            )

            print(f"[OK] {filename}")

        except Exception as e:
            # 記錄錯誤
            error_list.write(
                f"Line {idx+1}: {line.strip()}\nError: {str(e)}\n\n"
            )
            print(f"[ERROR] {filename} -> 已記錄並跳過")
            continue

    error_list.close()

from Taiwanese_Server_tl2ctl_phoneme_toolkit.tw_frontend import tw_frontend # 把tcp server上的工具直接拿來用 省的自己轉換
from en import english_cleaners2
def read_kasih_dir(kasih_path):
    """
    讀取 kasih 字卡的資料結構，僅用來合成字卡語音。
    """
    # print("read test")

    LANGUAGE = ['tai']

    frontend = tw_frontend(g2p_model="tw_tl")
    for root, dirs, files in os.walk(kasih_path):
        for file in files:
            # print(os.path.join(root, file))
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    for lang in LANGUAGE:
                        text_lang = lang
                        if text_lang == 'tai':
                            text_lang = 'tailo'
                        print(item[text_lang])
                        if (item[text_lang]):
                            print(item[f'{lang}_voice'])
                            if (lang == 'tai'):
                                result = frontend.get_phonemes(item[text_lang])[0]
                            elif (lang == 'en'):
                                result = english_cleaners2(item[text_lang].lower())
                            print(result)
                            result = " ".join(result)
                            # print(result)
                            synthesis(result, 28, 28, item[f'{lang}_voice'], False, "TW")
                        else:
                            print(item['zh_voice'], "無資料")
                            continue



if __name__ == '__main__':
    import argparse, sys

#   parser = argparse.ArgumentParser()
#   parser.add_argument("--text", default="kh0 o02 s0 e03")
#   parser.add_argument("--sid", type=int, default=55)
#   parser.add_argument("--lang", default='TW')
#   parser.add_argument("--en", action="store_true")
#   parser.add_argument("--sids", type=int, nargs='+')
#   parser.add_argument("--gen_text_lang", type=str)  # e.g., tw
#   args = parser.parse_args()

#   # 批次：直接呼叫你原本的 synthesis_file
#   if args.gen_text_lang:
#       lang = args.gen_text_lang.lower()   # 關鍵：傳入小寫以匹配 dict 與檔名
#       ids = args.sids if args.sids else [args.sid]
#       for sid in ids:
#           synthesis_file(language=lang, speaker_id=sid, net_g=net_g)
#       sys.exit(0)

#   # 單句，支援多 speaker
#   ids = args.sids if args.sids else [args.sid]
#   for sid in ids:
#       synthesis(args.text, sid, sid, f"default_{sid}.wav", args.en, args.lang)



    # 1111 偷改
    parser = argparse.ArgumentParser()
    # # text = "gn3 i33 n3 ai33 v3 ui33 v3 ooi33 th3 ung35" # 直接改下面synthesis即可 若要改text通常都吃最下面text
    text = "an22 ts2 oo22 tsch2 i21" # 早安
    text = "s0 ng02 p0 uann05" # 算盤
    text = "sil iau21 k2 e23"
    text = "ia22 l2 ioong22 s2 am21 gn2 it24 l2 ooi25 tsh2 oong25 th2 eu23 v2 ooi23 k2 am22 k2 ook24 t2 oo22 v2 i23 th2 ung23 h2 an25 k2 oo23 oo21 l2 i23 t2 u22"
    text = "tsh2 u21 tsh2 u21 tsh2 u21 oo21"
    text = "oo21 s2 ir22 tsh2 ut24 h2 iet24"
    text = "sil oo21 l2 i23"
    text = "sil gn3 i38 jr3 iu38 m3 a31 k3 ai35 sc3 im38 s3 ir33 h3 e33 m3 oo35 sil"
    text = "an24 ts2 oo22"
    text = "oo21 l2 i23 t2 u22 iu21 s2 ir25 v2 ooi23 oo21 sc2 ia23 p2 a21 iu21 s2 ir25 v2 ooi23 oo21 s2 ui22 sc2 ia23"
    text = "sil iu21 k2 ook24 ioong23"
    text = "th2 ai23 iook24 sil iu21"
    text = "m2 e23 h2 e23 it24 p2 au21 p2 au21 e22 t2 u23 ts2 oong21 t2 oo23 ts2 at28 p2 uk24 s2 irt28 l2 ook28"
    # text = "ts0 oong03 th0 ai03 dl0 oong01 iau02 k0 o02 ph0 ieen07 h0 ioong03"
    # # text = "k2 im21 p2 u21 gn2 it24 th2 ien21 h2 i23" # 今天天氣
    # # text = "k3 im38 p3 u38 gn3 it38 th3 ien38 h3 i35"
    # # text = "n25 h2 au23" # 妳好
    # # text = "an22 ts2 oo22 th2 ien21 h2 i23"
    parser.add_argument("--text", default=text)
    # # # parser.add_argument("--sid", default=342)     # sixsian
    # # # parser.add_argument("--sid", default=54)      # TW male
    # # # parser.add_argument("--sid", default=55)      # TW female 語速本身較快
    # # parser.add_argument("--sid", default=68)        # M04 盧哥
    # # # parser.add_argument("--sid", default=25)
    parser.add_argument("--sid", default=0)
    # parser.add_argument("--sid", default=1)
    # # # parser.add_argument("--sid", default=154)     # xf
    # # # parser.add_argument("--sid", default=161)     # zh
    # # # parser.add_argument("--sid", default=933)   # en
    # # # parser.add_argument("--sid", default=933)   # en
    # # # parser.add_argument("--sid", default=141)   # id
    # # # 348 22khz_male_trad
    # # # 349 22khz_female_trad
    # # # 382|aidatatang_200zh_G5693
    parser.add_argument("--lang", default='HAK')
    parser.add_argument("--en", default=False)
    args = parser.parse_args()
    result = text
    print(result)
    # synthesis("p4 uoo41", int(args.sid), int(args.sid), "波.wav", False, "TZH")
    # synthesis("kh0 o02 s0 e03", int(args.sid), int(args.sid), "khò-sè.wav", False, "TW")
    # synthesis("s0 ng02 p0 uann05", int(args.sid), int(args.sid), "正常句子.wav", False, "TW")
    # 1111 到這邊
    synthesis(result, int(args.sid), int(args.sid), "xmd1108-239test1.wav", False, "HAK")
    #synthesis(result, int(args.sid), int(args.sid), "91.wav", False, "HAK")
    # synthesis("n4 i43 h4 au43", int(args.sid), int(args.sid), "n4_i43_h4_au43.wav", False, "TZH")
    # synthesis(english_cleaners2("What's the weather today?"), int(args.sid), int(args.sid), "今天天氣怎麼樣.wav", False, "EN")
    # new_model, _, _, _ = utils.load_checkpoint(f"logs/tw_1211_trim_tail_coverage_test/{use_g_pth_file}.pth", net_g, None)
    # synthesis_file(language=args.lang, speaker_id=30, net_g=new_model)
    # synthesis("tschiann1 theh3 tshai2 tuann1 hoo3 gua1", 21, 21, "30_testfile.wav", False, "TW")
    # frontend = tw_frontend(g2p_model="tw_tl")
    # result = sixsian_frontend.get_phonemes(text)
    # test_string = "," + test_string + ","
    # test_string = test_string
    # test_string = frontend.get_phonemes(test_string, isChorus = False)[0]
    # print(test_string)
    # new_string = []
    # for i in range(len(test_string)):
    #     if "ng" in test_string[i]:
    #         new_string.append("u")
    #         new_string.append("inn")
    #     else:
    # test_string = ['q1 ', 'ing13', 'n1 ', 'i13', 'b1', 'u12', 'u12', 'u12', 'u12', 'u12', 'iao14', 'zh1 ', 'e14', 'iang14', 'z1 ', 'ii15']
    # result = " ".join(test_string)
    #         new_string.append(test_string[i].replace(" ", ""))
    # result = ", " + result + " ,"
    # result = "sil g0 ua01 b0 eh04 tsc0 iah03 p0 u07 inn07" # 我要吃飯
    # result = "g0 ua01 b0 eh04 tsc0 iah03 p0 ng07" # 我要吃飯
    # result = "sil tsc0 iann07 h0 u0 inn07" # 真遠
    # result = "k0 u05 inn07 m0 u05 inn05" # 關門
    # result = "sil k0 e07 n0 u07 inn07" # 雞蛋
    # result = "sil k0 e07 n0 ui07" #雞蛋
    # result = "sil s0 u07 inn07 s0 u01 inn01" # 酸酸
    # result = "h0 u07 inn07 h0 u03 inn03" # 遠遠
    # result = "kh0 a07 tsh0 u01 inn01" # 尻川
    # result = "t0 ng02 kh0 i03" # 其他腔調
    # result1 = "t0 u02 inn02 kh0 i03" # 轉去 宜蘭腔
    # result2 = "t0 u02 inn02 i0 i03" # 轉去 宜蘭腔
    # result = "uo13 iao14 q1 v14 m1 ai13 z1 ao13 c1 an11"
    # result = "ts0 ui01 k0 o02 ai02 kh0 au07 ph0 ue05 tsc0 iah08"
    # TZH
    # 1047|MA001-F01
    # 1048|MA001-M07

    # synthesis(result, 28, 28, "kasih_字卡.wav", False, "TW", )
    # synthesis(result, 24, 24, "kasih_字卡.wav", False, "TW")
    # synthesis(result1, 16, 16, "轉去_宜蘭_show.wav", False, "TW")
    # print(result2)
    # synthesis(result2, 16, 16, "轉去_宜蘭_省第二字聲母_show.wav", False, "TW")

    # test_string = "uah8-tong7"
    # test_string = " ".join(frontend.get_phonemes(test_string)[0])
    # synthesis("g0 ua01 b0 e03 h0 iau02 k0 oong02 t0 ai07 g0 i02", 16, 16, "testfile1.wav", False, "TW")
    # synthesis("ts0 ui01 k0 o02 a0 ai02 kh0 au07 ph0 ue05 tsc0 iah08", 16, 16, "testfile2.wav", False, "TW")
    # synthesis("ts0 ui01 k0 o02", 16, 16, "testfile3.wav", False, "TW")
    # synthesis("sc0 i07 k0 ue01", 16, 16, "testfile4.wav", False, "TW")


    # synthesis("b0 o05 a0 ai03", 16, 16, "合音1.wav", False, "TW")
    # synthesis("sil b0 o05 ai03", 16, 16, "合音2.wav", False, "TW") # 第二字為純韻母，第一字韻母去掉?
    # synthesis("b0 o05 i0 iau02 k0 in02", 16, 16, "合音3.wav", False, "TW")
    # synthesis("b0 ua05 k0 in02", 16, 16, "合音4.wav", False, "TW")
    # synthesis("b0 o07 e0 e07", 16, 16, "合音5.wav", False, "TW")
    # synthesis("b0 e0 e07", 16, 16, "合音6.wav", False, "TW") # 第二字為純韻母，第一字韻母去掉?
    # synthesis("ts0 a01 h0 ng01", 16, 16, "合音7.wav", False, "TW")
    # synthesis("ts0 a01 ng05", 16, 16, "合音8.wav", False, "TW") # 第二字有聲母，第二字聲母去掉?

    # synthesis("k0 a07 i0 i07 ph0 ah02 sc0 i02", 16, 16, "合音9.wav", False, "TW")
    # synthesis("k0 i0 i07 ph0 ah02 sc0 i02", 16, 16, "合音10.wav", False, "TW") # 第二字為純韻母，第一字韻母去掉？顯然不行
    # synthesis("k0 ai07 ph0 ah02 sc0 i02", 16, 16, "合音11.wav", False, "TW") # 第二字為純韻母，第二字聲母去掉，效果較好

    # synthesis("k0 a03 dl0 ang05 ph0 ah04", 16, 16, "合音12.wav", False, "TW")
    # synthesis("k0 a03 ang05 ph0 ah04", 16, 16, "合音13.wav", False, "TW") # 第二字有聲母，第二字聲母去掉?
    # synthesis("k0 a03 ang03 ph0 ah04", 16, 16, "合音14.wav", False, "TW")
    #synthesis("h0 oo07 dl0 ang05 ph0 ah04", 16, 16, "合音23.wav", False, "TW")
    #synthesis("h0 oo07 ang05 ph0 ah04", 16, 16, "合音24.wav", False, "TW")

    #synthesis("sc0 iann01 dl0 ang05 ph0 ah04", 16, 16, "合音34.wav", False, "TW")
    #synthesis("sc0 iann01 ang05 ph0 ah04", 16, 16, "合音35.wav", False, "TW")


    # synthesis("k0 dl0 ang07 ph0 ah02 sc0 i02", 16, 16, "合音14.wav", False, "TW") # 去掉第一字韻母顯然不行

    # synthesis("tsch0 iu01 p0 io01 a0 a01", 75, 75, "手錶.wav", False, "TW")
    # synthesis("tsch0 iu01 p0 io01 a0 a01", 16, 16, "手錶1.wav", False, "TW")

    # synthesis("sil tsch0 iu01 kh0 uan05", 75, 75, "手環.wav", False, "TW") # 去掉第一字韻母顯然不行
    # synthesis("sil tsh0 ui02 a0 am01", 75, 75, "口罩.wav", False, "TW") # 去掉第一字韻母顯然不行
    # synthesis("sil th0 eh03 i0 ioh07 a0 a02", 75, 75, "提藥仔 .wav", False, "TW") # 去掉第一字韻母顯然不行
    # synthesis("sil tsh0 ui02 tsch0 iu07 kh0 au07 a0 a02", 75, 75, "刮鬍刀.wav", False, "TW") # 去掉第一字韻母顯然不行



    # synthesis("kh0 i02 p0 ieen03 s0 oo02", 28, 28, "去廁所.wav", False, "TW")
    # synthesis("ph0 uah7 dl0 ieen07", 28, 28, "項鍊.wav", False, "TW")
    # synthesis("h0 oong07 k0 ieng02", 26, 26, "掃地機器人.wav", False, "TW")

    #synthesis("i0 im07 ts0 ui02", 75, 75, "淹水.wav", False, "TW") # 去掉第一字韻母顯然不行

    #synthesis("i0 im07 g0 ak04 k0 au02 sc0 iek04", 75, 75, "音樂教室.wav", False, "TW") # 去掉第一字韻母顯然不行

    #synthesis("b0 o07 h0 uat04 i0 i03", 75, 75, "無法伊.wav", False, "TW")
    #synthesis("b0 o07 h0 uat04 i03", 75, 75, "無法伊無append.wav", False, "TW")
    #synthesis("dl0 i01 b0 at04 i0 i01 b0 o03", 75, 75, "你認識他嗎_2.wav", False, "TW")
    # synthesis("tsc0 inn05 sc0 i07 i07 kh0 iam03 e03", 16, 16, "錢是他欠的.wav", False, "TW")
    # synthesis("sil u03 iann02 b0 o03", 16, 16, "有影無.wav", False, "TW")
    # synthesis("sil dl0 i01 u07 tsc0 iah03 p0 ng07 b0 o05 sil", 17, 17, "你有吃飯嗎_300_tune.wav", False, "TW")


    #synthesis("i0 i0 i0 i0 i0 i01", 75, 75, "母音測試.wav", False, "TW")



    # synthesis("dl0 i01 u0 u07 h0 uat08 sc0 io01 b0 o03", 16, 16, "testfile5.wav", False, "TW")
    # synthesis("k0 in07 a0 a01 djz0 it08 th0 inn07 kh0 i03 tsc0 in01 h0 o02", 16, 16, "testfile6.wav", False, "TW")
    # synthesis("sc0 im07 s0 oo01 a0 ai03 e0 e07 dl0 ang07", 16, 16, "testfile心所愛的人.wav", False, "TW")
    # synthesis("sc0 ioong07 sc0 im01 e0 e07 k0 ua01", 16, 16, "testfile傷心的歌.wav", False, "TW")
    # synthesis("k0 uai01 a0 a02", 16, 16, "testfile拐杖.wav", False, "TW")
    # synthesis("ng2 a21 sc2 im21 k2 oon21 t2 u22 ts2 oo23 sc2 im21 k2 oon21 t2 u22 m2 en22 sil m2 ien23 h2 oong23 h2 an25 m2 ang25 sc2 ioong22 t2 oo22 ooi23 gn2 ioong22 p2 an21 p2 eu22 tsch2 in25 sil", 16, 16, "testfile1Hak.wav", False, "HAK")
    # synthesis("ng25 ts2 oo23 m2 a22 k2 e23 ooi23 k2 ieu23 n2 oo22 sil l2 oo22 l2 iu21 s2 u23 sil", 16, 16, "testfile2Hak.wav", False, "HAK")
    # synthesis("hio2 khun3 tscit4 e7", 16, 16, "testfile3Hak.wav", False, "HAK")
    # synthesis("sil sil g0 ua01 b0 e03 h0 iau02 k0 oong02 t0 ai07 g0 i02", 24, 24, "testfile1sil.wav", False, "TW")
    # synthesis("sil sil ts0 ui01 k0 o02 a0 ai02 kh0 au07 ph0 ue05 tsc0 iah08", 24, 24, "testfile2sil.wav", False, "TW")
    # synthesis("sil sil ts0 ui01 k0 o02", 24, 24, "testfile3sil.wav", False, "TW")
    # synthesis("sil sil sc0 i07 k0 ue01", 24, 24, "testfile4sil.wav", False, "TW")
    # synthesis("sil sil dl0 i01 u0 u07 h0 uat08 sc0 io01 b0 o03", 24, 24, "testfile5sil.wav", False, "TW")
    # synthesis("sil sil k0 in07 a0 a01 djz0 it08 th0 inn07 kh0 i03 tsc0 in01 h0 o02", 24, 24, "testfile6sil.wav", False, "TW")

    # synthesis("ˈɪlnəs", 22, 22, "testfile_en_1.wav", False, "EN")
    # synthesis("sᵻvˈɪɹ ˈɪlnəs kˈɑːɹd", 22, 22, "testfile_en_2.wav", False, "EN")
    # synthesis("fˈɛstɪvəl sˌɛləbɹˈeɪʃən", 22, 22, "testfile_en_3.wav", False, "EN")
    # synthesis("pˈiːl", 22, 22, "testfile_en_4.wav", False, "EN")
    # synthesis(english_cleaners2("hot or cold water"), 2712, 2712, "testfile_en_5.wav", False, "EN")
    # synthesis(english_cleaners2("How much is this"), 2712, 2712, "testfile_en_6.wav", False, "EN")
    # synthesis(english_cleaners2("What local specialties do you have"), 2712, 2712, "testfile_en_7.wav", False, "EN")
    # synthesis(english_cleaners2("I would like to make a reservation"), 2712, 2712, "testfile_en_7.wav", False, "EN")


    # synthesis("sil g0 ua01 b0 e03 h0 iau02 k0 oong02 t0 ai07 g0 i02", 24, 24, "testfile1_single_sil.wav", False, "TW")
    # synthesis("sil ts0 ui01 k0 o02 a0 ai02 kh0 au07 ph0 ue05 tsc0 iah08", 24, 24, "testfile2_single_sil.wav", False, "TW")
    # synthesis("sil ts0 ui01 k0 o02", 24, 24, "testfile3_single_sil.wav", False, "TW")
    # synthesis("sil sc0 i07 k0 ue01", 24, 24, "testfile4_single_sil.wav", False, "TW")
    # synthesis("sil dl0 i01 u0 u07 h0 uat08 sc0 io01 b0 o03", 24, 24, "testfile5_single_sil.wav", False, "TW")
    # synthesis("sil k0 in07 a0 a01 djz0 it08 th0 inn07 kh0 i03 tsc0 in01 h0 o02", 24, 24, "testfile6_single_sil.wav", False, "TW")

    # synthesis("sc0 i07 k0 ue01 sc0 i07 k0 ue01 sc0 i07 sc0 i07", 24, 24, "西瓜西瓜西西.wav", False, "TW")
    # synthesis("sc0 i07 k0 ue01", 21, 21, "西瓜SHIKY.wav", False, "TW")
    #synthesis("sc0 k0 ue01", 24, 24, "西瓜1.wav", False, "TW")
    # synthesis("sil sil k0 iu02 h0 oo03 tsch0 ia01 sil", 24, 24, "救護車測試.wav", False, "TW")


    # print(synthesis("", 30, 30, "infer_test", False, TW))
    # spk_list = [6,7,31,56,57]
    # spk_list = [31, 56]
    #finetuned_model, _, _, _ = utils.load_checkpoint("logs/vietnamese_0816/G_192000.pth", net_g, None)
    # finetuned_model, _, _, _ = utils.load_checkpoint("logs/G_1000.pth", net_g, None)
    # finetuned_model, _, _, _ = utils.load_checkpoint("logs/tw_new_2/G_192000.pth", net_g, None)
    # for spk in spk_list:
        # synthesis_file(language=args.lang, speaker_id=spk, net_g=finetuned_model)

    # base_dir = f'gen_audio/56'
    # for f in os.listdir(base_dir):
    #     if not f.endswith('.wav'):
    #         os.remove(f'{base_dir}/{f}')
    #         continue
    #     vc(f'{base_dir}/{f}', finetuned_model)

    # read_kasih_dir("resource_for_kasih")

