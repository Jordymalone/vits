from typing import Dict, List, Tuple
import os
from symtable import symtable
from tl2ctl import askForService

import ch2tl
import tlsandhi

class tw_frontend():
    def __init__(self):
        self.punc = "：，；。？！“”‘’':,;.?!"


    def _get_initials_finals(self, sentence: str) -> Tuple[List[List[str]], bool]:
        initials = []
        finals = []

        sentence = sentence.replace("0", "")

        orig_initials, orig_finals = self._cut_vowel(sentence)
        for c, v in zip(orig_initials, orig_finals):
            if c and c not in self.punc:
                initials.append(c+'-')
            else:
                initials.append(c)
            if v not in self.punc:
                finals.append(v)
            else:
                finals.append(v)

        return initials, finals, True

    def _g2p(self, sentences: List[str]) -> Tuple[List[List[str]], bool]:
        phones_list = []

        initials, finals, status = self._get_initials_finals(sentences)
        if status == False:
            return [], False
        for c, v in zip(initials, finals):
            if c and c not in self.punc:
                phones_list.append(c)
            if c and c in self.punc:
                phones_list.append('sp')
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

    def get_phonemes(self, sentence: str) -> List[str]:
        phonemes, status = self._g2p(sentence)
        if status == False:
            return [], False
        print(phonemes)
        r = ''
        for p in phonemes:
            if not p.endswith('-'):
                r = f'{r}{p}'
            else:
                r = f'{r} {p}'
        r = r.strip()

        return r, True

if __name__ == "__main__":
    tw = tw_frontend()
    raw = ch2tl.askForService("今天吃早餐")
    print(raw)
    sandhi = tlsandhi.askForService(raw["tailuo"])
    print(sandhi)
    ctl = askForService(sandhi)
    print(ctl)
    pinyin = tw.get_phonemes(ctl)
    print(pinyin)
    from infer import synthesis

    # synthesis(pinyin[0], 183, 183, 'test.wav', None, 2)
    # with open(f'CTL.txt','r', encoding='utf-8') as f:
    #     lines = f.readlines()


    # file_name = 0
    # for each in lines:
    #     file, pinyin = each.split()
    #     pinyin = pinyin.replace(' ', '')
    #     # synthesis(text, 183, 183, filename, None, lang_map[lang])
    #     print(f'sytneshis for {file}, language: 2, pinyin: {pinyin}')
    #     pinyin = tw.get_phonemes(pinyin)[0]
    #     synthesis(pinyin, 183, 183, f'{file_name:004}_{file}.wav', None, 2)
    #     with open(f'output.txt','a',encoding='utf8') as f2:
    #         f2.write(f'{file_name:004}_{file}.wav\n')
    #     file_name+=1
