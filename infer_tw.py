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

speaker_dict = {}

def load_speaker_dict():
    with open(f'/home/p76111652/Linux_DATA/synthesis/corpus/22050/dataset/mixed_5_id.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip().split("|")
            speaker_dict[line[1]] = line[0]

def process_text(input_text, english_flag):
    input_text = input_text.upper()
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

hps = utils.get_hparams_from_file("logs/tw_test_char/config.json")

def get_symbols():
    _pad        = '_'
    _punctuation = ';:,.!?¡¿—…-–"«»“” '
    _tone = '0123456789'
    # 240816 竟烽要留的
    _sym = [' ', '!', ',', '.', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̩', 'θ', 'ᵻ']

   # _sym = [' ', '', "'", ',', '.', '1', '2', '3', '4', '5', '6', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ô', 'ă', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɣ', 'ɤ', 'ɪ', 'ɯ', 'ɲ', 'ʂ', 'ʃ', 'ʈ', 'ʊ', 'ʐ', 'ʧ', 'ʰ', 'ʷ', 'ˈ', 'ˌ', '̆', '͡', 'ầ']
    with open(f'filelists/tw_test_2/lang_phones.txt','r',encoding='utf-8') as f:
        phonemes = f.readlines()
        phonemes = [p.strip() for p in phonemes]

    # Export all symbols:
    # symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_tone) + list(_extra_indo) + list(_special)
    symbols = list(phonemes) + [_pad] + list(_punctuation) + list(_tone)
    base_offset = len(symbols)
    sym_offset = len(_sym)
    symbols += _sym * 2
    SPACE_ID = symbols.index(" ")
    return symbols

# symbols = get_symbols()
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("logs/tw_test_char/G_504000.pth", net_g, None)

def synthesis(text, speaker_id, speaker_name, filename, english_flag, langauge):
    processed_text = process_text(text, english_flag)
    result_np_arr = []
    start_time = time.time()
    processed_text = processed_text.replace(", ",",")
    for each in str(processed_text).split(","):
        # prevent None
        if not each:
            continue
        each = each.strip()
        stn_tst = get_text(each, langauge)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([int(speaker_id)]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.3, noise_scale_w=0.2, length_scale=1.0)[0][0,0].data.cpu().float().numpy()
            result_np_arr.append(audio)

    concatenated_audio = np.concatenate(result_np_arr)
    elapsed_time = time.time() - start_time
    print(f"Synthesis for speaker {speaker_id} took {elapsed_time} seconds.")
    if not os.path.exists(f'./gen_audio'):
        os.makedirs(f'./gen_audio')
    if not os.path.exists(f'./gen_audio/{speaker_name}'):
        os.makedirs(f'./gen_audio/{speaker_name}')

    audio_path = f"./gen_audio/{speaker_name}/{filename}"
    sf.write(audio_path, concatenated_audio, 48000)
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
    sid_src = torch.LongTensor([56]).cuda()
    target_speakerId = 31

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    text = ""
    parser.add_argument("--text", default=text)
    parser.add_argument("--sid", default=0)
    parser.add_argument("--lang", default='tw')
    parser.add_argument("--en", default=False)
    args = parser.parse_args()
    # spk_list = [6,7,31,56,57]
    spk_list = [0,1,2]
    #finetuned_model, _, _, _ = utils.load_checkpoint("logs/vietnamese_0816/G_192000.pth", net_g, None)
    # finetuned_model, _, _, _ = utils.load_checkpoint("logs/G_1000.pth", net_g, None)
    finetuned_model, _, _, _ = utils.load_checkpoint("logs/tw_test_char/G_504000.pth", net_g, None)
    for spk in spk_list:
        synthesis_file(language=args.lang, speaker_id=spk, net_g=finetuned_model)

    # base_dir = f'gen_audio/56'
    # for f in os.listdir(base_dir):
    #     if not f.endswith('.wav'):
    #         os.remove(f'{base_dir}/{f}')
    #         continue
    #     vc(f'{base_dir}/{f}', finetuned_model)

