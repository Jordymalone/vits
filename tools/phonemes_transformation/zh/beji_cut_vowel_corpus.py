from pypinyin import lazy_pinyin, Style, load_phrases_dict, load_single_dict
from typing import Any, List, Tuple
import re
from opencc import OpenCC
import json
import sys
import os

# 如果有自定义模块，如CKIP、number_normalization，请确保它们在相应的路径中
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../../')
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}')

# from . import CKIP, number_normalization # 用於package中
import CKIP # 用於單獨運行
import number_normalization

class zh_frontend():
    def __init__(self) -> None:
        self.punc = "：，；。？！“”‘’':,;.?!"
        self.silence_control = '~'
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 指定字典文件的路径
        json_file_path = os.path.join(current_dir, 'rb_dict.json')
        # polyphone_file_path = os.path.join(current_dir, 'polyphone_dict.json')
        load_phrases_dict_path = os.path.join(current_dir, 'load_phrases_dict.json')

        # # 原本使用 polyphone_dict 自行替換，後改為使用 load_phrases_dict ，pypinyin 提供的字典更換
        with open(load_phrases_dict_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        load_phrases_dict(data)

        load_phrases_dict({'頂著': [['ding3'], ['zhe5']]})
        # 針對單個字更改讀音，另外特別提醒，"崖"字在中國拼音系統中完全沒有相應讀音，如有需要精確發音訓練，只能在最後替換為"iai2"
        # 此項可以替代 rb_dict，待完成。
        # load_single_dict({ord('拎'): 'ling1'})  # 調整單個字發音

        # 从JSON文件读取rb_dict
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.rb_dict = json.load(f)

        # with open(polyphone_file_path, 'r', encoding='utf-8') as f:
        #     self.polyphone_dict = json.load(f)

    def to_half_width(self, text):
        converted_text = ''
        for char in text:
            ascii_code = ord(char)
            # 如果是全角字符
            if 65281 <= ascii_code <= 65374:
                converted_char = chr(ascii_code - 65248)  # 转换为半角
            else:
                converted_char = char
            converted_text += converted_char
        return converted_text

    def _get_initials_finals(self, word: str) -> Tuple[List[str], List[str], bool]:
        initials = []
        finals = []
        print(f"_get_initials_finals 收到的: {word}")
        try:
            word = word.replace("嗯", "恩")
            word = self.to_half_width(word.strip())
            orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
            orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            print(orig_initials,orig_finals)

            # 处理i、ii、iii的情况
            for c, v in zip(orig_initials, orig_finals):
                if re.match(r'i\d', v):
                    if c in ['z', 'c', 's']:
                        v = re.sub('i', 'ii', v)
                    elif c in ['zh', 'ch', 'sh', 'r']:
                        v = re.sub('i', 'iii', v)
                c = c.strip()
                v = v.strip()
                if c == v:
                    if not len(c) == 1:
                        for char in c:
                            initials.append(char)
                            finals.append(char)
                        continue
                    if c in self.punc and v in self.punc:
                        initials.append(c)
                        finals.append(v)
                        continue

                if c and c not in self.punc:
                    initials.append(c+'1')
                else:
                    initials.append(c)
                if v not in self.punc:
                    finals.append(v[:-1]+'1'+v[-1])
                else:
                    finals.append(v)

            return initials, finals, True

        except:
            return [], [], False

    def _g2p(self, sentences: List[str]) -> Tuple[List[List[str]], bool]:
        if len(sentences) == 1:
            if sentences[0] == '~':
                return [['~']], True

        phones_list = []
        foreign_sentences = []

        for seg in sentences:
            phones = []
            initials = []
            finals = []

            # 使用 CKIP 進行分詞
            seg_cut = CKIP.call_ckip([seg])
            for word, pos in seg_cut:
                # 如果檢測到外語（FW）
                if pos == 'FW':
                    print(f"檢測到外語: {word}，請注意，英文會在製作語料時被清除，但是音檔方面仍然會有該英文詞彙")
                    foreign_sentences.append(seg)  # 將外語詞彙加入列表中
                    continue
                # sub_initials, sub_finals, status = self._get_initials_finals(word)
                # if status == False:
                #     return [], False
                # initials.extend(sub_initials)
                # finals.extend(sub_finals)

            # 將外語句子寫入檔案
            if foreign_sentences:
                with open("含有外語句子.txt", "w", encoding="utf-8") as f:
                    for sentence in foreign_sentences:
                        f.write(sentence + "\n")

            # 去除英文字符
            seg = re.sub('[a-zA-Z]+', '', seg)

            initials, finals, status = self._get_initials_finals(seg)



            print(initials)
            for c, v in zip(initials, finals):
                if c and c not in self.punc:
                    phones.append(c)
                if c and c in self.punc:
                    phones.append('sp')
                if c and c in self.silence_control:
                    phones.append('0.1s')
                if v and v not in self.punc:
                    phones.append(v)

            phones_list.append(phones)
        print(f"phones_list:{phones_list}")
        return phones_list, True

    def get_phonemes(self, sentence: str) -> Tuple[List[str], bool]:
        # 将句子中的数字和日期转换为中文格式
        sentences = [number_normalization.askForService(language='ch', chinese=sentence)]

        # 将简体中文转换为繁体中文，避免词语转换
        cc = OpenCC('s2tw')
        for _index, _sentence in enumerate(sentences):
            sentences[_index] = cc.convert(_sentence)

        # 根据字典替换一些特定的词
        for _index, _sentence in enumerate(sentences):
            for key, value in self.rb_dict.items():
                sentences[_index] = sentences[_index].replace(key, value)

        # 获取句子的拼音
        phonemes, status = self._g2p(sentences)

        # 复制拼音以进行替换操作
        polyphone_correction_phonemes = phonemes.copy()

        # 遍历每个句子，检查是否需要进行多音字的替换
        # for _index, _sentence in enumerate(sentences):
        #     for key, value in self.polyphone_dict.items():
        #         if key in sentences[_index]:
        #             # 如果句子中有多音字，获取错误的拼音
        #             wrong_phonemes, _ = self._g2p([key])
        #             # 将错误的拼音替换为正确的拼音
        #             polyphone_correction_phonemes[0] = self._replace_section(
        #                 polyphone_correction_phonemes[0], wrong_phonemes[0], value
        #             )

        # 如果G2P转换失败，返回空结果
        if not status:
            return [], False

        return " ".join(polyphone_correction_phonemes[0]), True

    def _replace_section(self, lst, target_section, new_section):
        """
        把原列表中的某一段（target_section）完全替换成新的区段（new_section），无论长度是否相同。

        :param lst: 原始列表
        :param target_section: 需要被替换的区段
        :param new_section: 替换成的区段
        :return: 修改后的列表
        """
        # 找到需要被替换区段的起始索引
        start_index = -1
        for i in range(len(lst) - len(target_section) + 1):
            if lst[i:i + len(target_section)] == target_section:
                start_index = i
                break

        # 如果找到匹配的区段，进行替换
        if start_index != -1:
            end_index = start_index + len(target_section)
            # 返回替换后的结果
            return lst[:start_index] + new_section + lst[end_index:]
        else:
            # 如果没找到匹配区段，返回原列表
            return lst

if __name__ == "__main__":

    frontend = zh_frontend()
    sentence = '拎著bag血包頂著'
    phonemes, status = frontend.get_phonemes(sentence)
    if status:
        print(phonemes)
    else:
        print("拼音转换失败")