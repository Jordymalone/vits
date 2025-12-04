""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols
from text.symbols import _sym, base_offset, sym_offset


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names, langauge):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  for char in text:
    if char == '':
      continue
    symbol_id = symbols.index(char)
    sequence.append(symbol_id)
  return sequence


def cleaned_text_to_sequence(cleaned_text, language):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  cleaned_text = cleaned_text.strip().replace("  ", " ")
  # print("now in text/__init__.py's cleaned_text_to_sequence")
  # phonemes_lang = ['ZH', 'TW', 'TZH', 'HAK', 'JP']
  phonemes_lang = ['ZH', 'TZH', 'HAK', 'JP', 'TW'] # 用來判斷是音素還是chracter 發音
  offset = {
    'EN' : 0,
    'ID' : 1,
    'VI' : 2
  }
  # offset = {
  #   'VI' : 0,
  # }
  
  if not language in phonemes_lang:
    for ph in cleaned_text: # 竟鋒寫給 chracter infer 用的
      if ph == '':
        raise Exception(f'Found empty string!, ph: {cleaned_text}')
      try:
        sequence.append(_sym.index(ph) +  sym_offset * offset[language] + base_offset) 
      except ValueError:
        if not ph == ' ':
          print(f"ValueError: Symbol '{ph}' not found in _sym, now cleaned_text: {cleaned_text}")
        # print(f"now language: {language}, now cleaned_text: {cleaned_text}")
  else:
    for ph in cleaned_text.split(' '):
      ph = ph.strip()
      try:
        sequence.append(_symbol_to_id[ph])
      except:
        # raise Exception(f'Found empty string! phoneme {cleaned_text}, {ph} not found')
        raise Exception(f'Symbol not found! Original phoneme string: "{cleaned_text}", problematic segment: "{ph}" not found in symbol_to_id map.')
  # if len(sequence) != len(cleaned_text): # 似乎也是為了 chracter trainig 寫的
  #   print("cleaned_text :" , cleaned_text)
  #   print("sequence :", sequence)
  #   raise Exception(f'Length of sequence {len(sequence)} and cleaned_text {len(cleaned_text)} does not match')
  return sequence


def sequence_to_cleaned_text(sequence, language, symbols):
    '''Converts a sequence of IDs to a string of text corresponding to the symbols.
    Args:
        sequence: list of integers representing the sequence
        language: integer representing the language offset
        symbols: list of symbols used in the conversion
    Returns:
        String corresponding to the sequence of IDs
    '''
    cleaned_text = ''
    base_offset = len(symbols)
    for symbol_id in sequence:
        if symbol_id % base_offset == 0 and symbol_id != 0:
            raise Exception('Invalid symbol ID!')
        char_index = (symbol_id - language * base_offset) % base_offset
        cleaned_text += symbols[char_index]
    return cleaned_text


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
