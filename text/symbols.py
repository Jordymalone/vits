""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''

_pad        = '_'
_punctuation = ';:,.!?¡¿—…-–"«»“” '
_tone = '0123456789'
# 240816 竟烽要留的
# _sym = [' ', '!', ',', '.', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̩', 'θ', 'ᵻ']
# 20240821 for taiwanese character
# _sym = [' ', '!', ',', '.', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̩', 'θ', 'ᵻ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，']
# 20240821 越南文
# _sym = [' ', '', "'", ',', '.', '1', '2', '3', '4', '5', '6', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ô', 'ă', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɣ', 'ɤ', 'ɪ', 'ɯ', 'ɲ', 'ʂ', 'ʃ', 'ʈ', 'ʊ', 'ʐ', 'ʧ', 'ʰ', 'ʷ', 'ˈ', 'ˌ', '̆', '͡', 'ầ']
# 241104 景霈訓練單台語用的
# _sym = ['', ',', '0', '1', '2', '3', '4', '5', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'y', 'z']
# 241106 景霈訓練雙母音+nycu語者台語

# 3834_5_retrainpaintako_vad
# _sym = [
#     ' ', '!', '"', ',', '.', ':', ';', '?',
#     'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm',
#     'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z',
#     'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ',
#     'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʲ',
#     'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', 'ᵻ'
# ]

# 若沒有任何 char 就用這個
_sym = []


# 250109 景霈訓練純英語
# _sym = ['', '!', '"', ',', '.', ':', ';', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʲ', 'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', 'ᵻ']

# 3646_vad_25_920
# _sym = [
#     ' ', '!', '"', ',', '.', ':', ';', '?',
#     'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm',
#     'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z',
#     'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ',
#     'ɡ', 'ɪ', 'ɬ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʲ',
#     'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', 'ᵻ'
# ]

# with open(f'filelists/Hakka_xf/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/Hakka_xf_vad/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/retraintw/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/double_phoneme_zh_tw/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/phonetic_test_zh/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/3646_vad_25_920/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/Hakka_xf/lang_phones.txt','r',encoding='utf-8') as f:
with open(f'filelists/Hakka_xm/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/hakka_six_v1/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/hakka_wo_hac/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/hakka_six_segment_4096/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/Hakka_hf/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/Hakka_hm/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'filelists/tw_1220_kaldi_300_noise/lang_phones.txt','r',encoding='utf-8') as f:
# with open(f'text/lang_phones.txt','r',encoding='utf-8') as f:
  phonemes = f.readlines()
phonemes = [p.strip() for p in phonemes]

# Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_tone) + list(_extra_indo) + list(_special)
symbols = list(phonemes) + [_pad] + list(_punctuation) + list(_tone)
base_offset = len(symbols)
sym_offset = len(_sym)
# symbols += _sym * 3
symbols += _sym * 2
SPACE_ID = symbols.index(" ")


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# def cleaned_text_to_sequence(cleaned_text, language):
#   '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
#     Args:
#       text: string to convert to a sequence
#     Returns:
#       List of integers corresponding to the symbols in the text
#   '''
#   sequence = []
#   phonemes_lang = ['ZH', 'TW', 'TZH', 'HAK']
#   offset = {
#     'EN' : 0,
#     'ID' : 1,
#   }
#   if not language in phonemes_lang:
#     for ph in cleaned_text:
#       if ph == '':
#         raise Exception('Found empty string!')
#       sequence.append(_sym.index(ph) +  sym_offset * offset[language] + base_offset)
#   else:
#     for ph in cleaned_text.split(' '):
#       if ph == '':
#         raise Exception('Found empty string!')
#       sequence.append(_symbol_to_id[ph])
#   return sequence

def cleaned_text_to_sequence(cleaned_text, language):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  cleaned_text = cleaned_text.strip().replace("  ", " ")
  # phonemes_lang = ['ZH', 'TW', 'TZH', 'HAK', 'JP']
  print("now in symbols.py's cleaned_text_to_sequence")
  phonemes_lang = ['ZH', 'TZH', 'HAK', 'JP', 'TW']
  offset = {
    'EN' : 0,
    'ID' : 1,
    'VI' : 2
  }
  # offset = {
  #   'VI' : 0
  # }
  if not language in phonemes_lang:
    for ph in cleaned_text:
      if ph == '':
        raise Exception(f'Found empty string!, ph: {cleaned_text}')
      print(_sym.index(ph))
      sequence.append(_sym.index(ph) +  sym_offset * offset[language] + base_offset)
  else:
    for ph in cleaned_text.split(' '):
      ph = ph.strip()
      try:
        sequence.append(_symbol_to_id[ph])
      except:
        raise Exception(f'Found empty string! phoneme {cleaned_text}, {ph} not found')

  return sequence


if __name__ == "__main__":
  text = "tsch2 iu23 tsch2 ioong23 h2 e23 it24 k2 e23 iu23 n2 un23 sil t2 et24 gn2 in25 sc2 iak24 iu23 m2 oo25 an22 k2 ien22 t2 an21 v2 a22 kh2 iun21 k2 e23 s2 irp28 tsch2 it24 s2 e23 s2 e23 a21 m2 ooi23 sil"
  text = "khia8 ti7 dleh3 sciek2 gua8 e3"
  print(cleaned_text_to_sequence(text, 'TW'))
