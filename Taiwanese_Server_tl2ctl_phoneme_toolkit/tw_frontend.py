from typing import Dict, List, Tuple

# processing pipeline
from .phoneme_process.number_normalization import askForService as num_nor
from .phoneme_process.ch2tl import askForService as ch2tl
from .phoneme_process.sandhi_client import askForService as sandhi
from .phoneme_process.tl2ctl_client import askForService as tl2ctl

import sys
import os
import re
import json
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../../')
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}')
# from logs.service_logger import service_logger

class tw_frontend():
    def __init__(self, g2p_model: str):
        self.punc = "：，；。？！“”‘’':,;.?!"
        self.g2p_model = g2p_model
        # self.logger = service_logger()
        # 獲取當前目錄
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 指定 rb_dict.json 的路徑
        bun_peh_path = os.path.join(current_dir, "台語文白異音字典.json")

        with open(bun_peh_path, 'r', encoding='utf-8') as f:
            self.bun_peh_dict = json.load(f)

    def _get_initials_finals(self, sentence: str, isChorus = False) -> Tuple[List[List[str]], bool]:
        initials = []
        finals = []
        tl_sandhi = sentence.replace("。", "，")
        # print(f'using: {self.g2p_model}')
        if tl_sandhi == "":
            # self.logger.info(f'TL_sandhi {sentence} empty, return empty list', extra={"ipaddr":""})
            return [], [], True
        try:
            if self.g2p_model == "tw":
                sentence = sentence.replace("麭", "紡") # 目前這字字典都沒有，先這樣處理 2024/10/28

                sentence = num_nor(language='tl', chinese=sentence)
                # self.logger.info(f'Number normalization: {sentence}', extra={"ipaddr":""})

                tl_text = ch2tl(text=sentence) # 無論是否有文白指定，先轉一次符合後續所有tl_text需求，如果有再處理並對word 覆蓋

                if '<文>' in sentence or '<白>' in sentence:
                    # 如果有強制文白標記，則此處進行轉換
                    # sign_word 是被標記的字list
                    bun_peh_list = []
                    print("split_by_tags_with_default: ", sentence, ",type:" , type(sentence))

                    sign_word = self.split_by_tags_with_default(str(sentence))
                    print("sign_word: ", sign_word)
                    for index,content in enumerate(sign_word):
                        if content[0] == '文':
                            for bun_peh_content in self.bun_peh_dict[content[1]]:
                                if bun_peh_content["文白"] == "文":
                                    bun_peh_list.append(self.convert_tone(bun_peh_content["音讀"]))
                                    continue # 可能有多個，目前先取第一個
                        elif content[0] == '白':
                            for bun_peh_content in self.bun_peh_dict[content[1]]:
                                if bun_peh_content["文白"] == "白":
                                    bun_peh_list.append(self.convert_tone(bun_peh_content["音讀"]))
                                    continue # 可能有多個，目前先取第一個
                        else :
                            tl_text = ch2tl(text=content[1])
                            bun_peh_list.append(tl_text['tailuo'])
                    print("bun_peh_list: ", bun_peh_list)
                    word = ' '.join(bun_peh_list)
                else: # 沒有標記，整句送入
                    word = tl_text['tailuo']


                # self.logger.info(f'Chinese to Taiwanese: {word}', extra={"ipaddr":""})
                tl_sandhi = sandhi(tai_luo= word)
                # self.logger.info(f'Sandhi from chinese words: {tl_sandhi}', extra={"ipaddr":""})
                if tl_sandhi == "":
                    # self.logger.error(f'TL_sandhi {sentence} got problem', extra={"ipaddr":""})
                    return [], [], True

                if tl_text['tailuo'] == 'Exceptions occurs':
                    # self.logger.error(f'Exceptions occurs, {sentence} got problem', extra={"ipaddr":""})
                    return [], [], False

            if self.g2p_model == "tw_tl_none":
                tl_sandhi = tl_sandhi
                # self.logger.info(f'Input already tone sandhied: {tl_sandhi}', extra={"ipaddr":""})

            if self.g2p_model == "tw_tl":
                tl_sandhi = sandhi(tai_luo=tl_sandhi)
                # self.logger.info(f'Sandhi from tw_tl: {tl_sandhi}', extra={"ipaddr":""})
        except:
            # self.logger.error(f'ch2phoneme pipline failed, {sentence} got problem', extra={"ipaddr":""}, exc_info=True)
            return [], [], False

        if '-' in tl_sandhi:
            tl_sandhi = tl_sandhi.replace('-', ' ')
        if tl_sandhi == "":
            # self.logger.error(f'TL_sandhi empty, skipping', extra={"ipaddr":""})
            return [], [], True
        ctl_text = tl2ctl(tai_luo=tl_sandhi)
        # self.logger.info(f'Taiwanese to CTL: {ctl_text}', extra={"ipaddr":""})
        orig_initials, orig_finals = self._cut_vowel(ctl_text)
        for c, v in zip(orig_initials, orig_finals):
            if c and c not in self.punc:
                initials.append(c+'0')
            elif not c: # 嘗試添加母音開頭，以避免母音爭奪其他字音 2024/11/06
                initials.append(v[0]+'0')
            else:
                initials.append(c)
            if v not in self.punc:
                finals.append(v[:-1]+'0'+v[-1])
            else:
                finals.append(v)
        # ====== ★ 新增的「合音省頭音」規則 ★ ======
        # 若 isChorus=True，且確定只有兩個音節，
        # 就把「第二個音節」的起首子音整個拿掉
        if isChorus and len(initials) >= 2:
            # 第二個音節的 initials 按需求直接清空
            initials[1] = ''            # 代表不放任何 C，只留下 V

        return initials, finals, True

    def _g2p(self, sentences: List[str], isChorus = False) -> Tuple[List[List[str]], bool]:
        if len(sentences) == 1:
            if sentences[0] == '~':
                return ['~'] , True
        phones_list = []

        initials, finals, status = self._get_initials_finals(sentences, isChorus)
        if status == False:
            return [], False
        for c, v in zip(initials, finals):
            if c and c not in self.punc:
                phones_list.append(c)
            if c and c in self.punc:
                phones_list.append('sil')
            if v and v not in self.punc:
                phones_list.append(v)

        return phones_list, True

    def _cut_vowel(self, sentence):
        vowel_list = ['a', 'e', 'i', 'o', 'u']
        initials = []
        finals = []
        flag = True
        word_lst = sentence.split()
        for word in word_lst:
            if word in self.punc:
                initials.append(word)
                finals.append('')

            for i, char in enumerate(word):
                if char in vowel_list:
                    initials.append(word[: i].strip())
                    finals.append(word[i :].strip())
                    flag = False
                    break
            if flag:
                for i, char in enumerate(word):
                    if char in ['m', 'n']:
                        initials.append(word[: i].strip())
                        finals.append(word[i :].strip())
                        flag = False
                        break
            flag = True

        return initials, finals

    def get_phonemes(self, sentence: str, isChorus = False) -> List[str]:
        phonemes, status = self._g2p(sentence, isChorus)
        if status == False:
            # self.logger.error(f'Error transforming: {sentence}, result: {phonemes}', extra={"ipaddr":""})
            return [], False
        # self.logger.info(f'Converting {sentence} to phonemes: {phonemes}', extra={"ipaddr":""})
        return phonemes, True


    def convert_tone(self, pinyin):
        # 定義所有聲調符號對應的數字轉換規則
        tone_mapping = {
            'ā': ('a', '1'), 'á': ('a', '2'), 'ǎ': ('a', '3'), 'à': ('a', '4'), 'a̍': ('a', '5'), 'a̋': ('a', '6'), 'à': ('a', '7'), 'â': ('a', '8'),
            'ē': ('e', '1'), 'é': ('e', '2'), 'ě': ('e', '3'), 'è': ('e', '4'), 'e̍': ('e', '5'), 'e̋': ('e', '6'), 'è': ('e', '7'), 'ê': ('e', '8'),
            'ī': ('i', '1'), 'í': ('i', '2'), 'ǐ': ('i', '3'), 'ì': ('i', '4'), 'i̍': ('i', '5'), 'i̋': ('i', '6'), 'ì': ('i', '7'), 'î': ('i', '8'),
            'ō': ('o', '1'), 'ó': ('o', '2'), 'ǒ': ('o', '3'), 'ò': ('o', '4'), 'o̍': ('o', '5'), 'ő': ('o', '6'), 'ò': ('o', '7'), 'ô': ('o', '8'),
            'ū': ('u', '1'), 'ú': ('u', '2'), 'ǔ': ('u', '3'), 'ù': ('u', '4'), 'u̍': ('u', '5'), 'ű': ('u', '6'), 'ù': ('u', '7'), 'û': ('u', '8'),
            'ṽ': ('v', '1'), 'v́': ('v', '2'), 'v̌': ('v', '3'), 'v̀': ('v', '4'), 'v̍': ('v', '5'), 'v̋': ('v', '6'), 'v̀': ('v', '7'), 'v̂': ('v', '8'),
        }

        # 遍歷所有符號替換它們
        for tone_vowel, (replacement_vowel, tone) in tone_mapping.items():
            if tone_vowel in pinyin:
                # 用普通母音替代聲調母音，並加上聲調數字
                pinyin = pinyin.replace(tone_vowel, replacement_vowel) + tone
                break  # 假設一個拼音中只會有一個帶調母音

        return pinyin

    def split_by_tags_with_default(self, text):
        print("curr type: ", type(text))
        # 定義正則表達式來匹配 <文> 或 <白> 的標記
        pattern = r"(<文>|</文>|<白>|</白>)"

        # 使用正則表達式將文字根據標記分割
        split_text = re.split(pattern, text)

        result = []
        current_label = "default"  # 預設為 default 標記

        # 遍歷分割後的段落，並依據標記分類
        for segment in split_text:
            # 忽略空白段落
            if not segment.strip():
                continue
            if segment == "<文>":
                current_label = "文"
            elif segment == "<白>":
                current_label = "白"
            elif segment == "</文>" or segment == "</白>":
                current_label = "default"
            else:
                # 添加標記和對應的內容到結果中
                result.append((current_label, segment.strip()))

        return result

if __name__ == "__main__":
    frontend = tw_frontend(g2p_model="tw_tl")
    result = frontend.get_phonemes("hioh4-khun3 tsit8-e7")
    print(" ".join(result[0]))
