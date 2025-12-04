import argparse
import torch
import commons
import utils
from models import SynthesizerTrn
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

import json

speaker_dict = {}

def load_speaker_dict():
    # with open(f'/home/p76111652/Linux_DATA/synthesis/corpus/22050/dataset/mixed_5_id.txt', 'r') as f:
    # with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/tw_1115_doublevVowel/mixed_5_id.txt', 'r') as f:
    with open(f'/home/p76131482/Linux_DATA/model/vits/filelists/tw_1115_doublevVowel/mixed_5_id.txt', 'r') as f:

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
hps = utils.get_hparams_from_file("logs/tw_1115_doublevVowel/config.json")

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
    _sym = []
    # 250109 景霈訓練純英語
    # _sym = ['', '!', '"', ',', '.', ':', ';', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʲ', 'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', 'ᵻ']

    with open(f'filelists/tw_1115_doublevVowel/lang_phones.txt','r',encoding='utf-8') as f:
        phonemes = f.readlines()
        phonemes = [p.strip() for p in phonemes]

    # Export all symbols:
    # symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_tone) + list(_extra_indo) + list(_special)
    symbols = list(phonemes) + [_pad] + list(_punctuation) + list(_tone)
    base_offset = len(symbols)
    sym_offset = len(_sym)
    symbols += _sym * 3
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
use_g_pth_file = 'G_287000'
# use_g_pth_file = 'G_290000'
# _ = utils.load_checkpoint("logs/tw_new_2/G_568000.pth", net_g, None) # 應該是之前測試別的專案寫死的
_ = utils.load_checkpoint(f"logs/tw_1115_doublevVowel/{use_g_pth_file}.pth", net_g, None)

def save_attn_txt(attn_tensor, save_path):
    attn = attn_tensor[0, 0].cpu().numpy()  # [T_audio, T_text]
    np.savetxt(save_path, attn, fmt="%.4f", delimiter=",")

def synthesis(text, speaker_id, speaker_name, filename, english_flag, language):
    # 處理輸入文字，直接作為一段
    processed_text = process_text(text, english_flag).strip()
    segments = [processed_text]
    result_np_arr = []
    start_time = time.time()

    # 對每段文字進行推理
    for each in segments:
        stn_tst = get_text(each, language)
        with torch.no_grad():
            x = stn_tst.cuda().unsqueeze(0)
            x_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([int(speaker_id)]).cuda()
            o, attn, _, _ = net_g.infer(
                x,
                x_lengths,
                sid=sid,
                noise_scale=0.0,
                noise_scale_w=0.1,
                length_scale=1.2
            )
            audio = o[0, 0].data.cpu().float().numpy()
            result_np_arr.append(audio)

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


def synthesis_file(language, speaker_id, net_g):
    with open(f'gen_text/{language}.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()

    language_dict = {
        'zh': 'ZH',
        'ctl': 'TZH',
        'ha': 'HAK',
        'tw': 'TW',
        'en': 'EN',
        'id': 'ID',
        'jp' : 'JP',
        'vi' : 'VI'
    }

    for line in lines:
        filename, text = line.strip().split("|")
        audio_path = synthesis(text, speaker_id, speaker_id, filename, False, language_dict[language])
        # vc(audio_path, net_g)

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
    parser = argparse.ArgumentParser()
    text = "瑜伽課"
    parser.add_argument("--text", default=text)
    parser.add_argument("--sid", default=0)
    parser.add_argument("--lang", default='zh')
    parser.add_argument("--en", default=False)
    args = parser.parse_args()
    # new_model, _, _, _ = utils.load_checkpoint(f"logs/tw_1211_trim_tail_coverage_test/{use_g_pth_file}.pth", net_g, None)
    # synthesis_file(language=args.lang, speaker_id=30, net_g=new_model)
    # synthesis("tschiann1 theh3 tshai2 tuann1 hoo3 gua1", 21, 21, "30_testfile.wav", False, "TW")
        # frontend = tw_frontend(g2p_model="tw_tl")
        # test_string = "瑜伽課"
        # test_string = test_string
        # test_string = frontend.get_phonemes(test_string, isChorus = False)[0]
    # print(test_string)
    # new_string = []
    # for i in range(len(test_string)):
    #     if "ng" in test_string[i]:
    #         new_string.append("u")
    #         new_string.append("inn")
    #     else:
    #         new_string.append(test_string[i].replace(" ", ""))
    # result = " ".join(test_string)
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
    # result = "," + result + ","
    # result = "h0 o01 kh0 i01 tsh0 ng05 a0 ah07"
    # print(result)
    synthesis(result, 28, 28, "kasih_字卡.wav", False, "ZH", )
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

