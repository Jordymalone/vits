# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from typing import Tuple

import jieba
from pypinyin import lazy_pinyin
from pypinyin import Style


class ToneSandhi():
    def __init__(self):
        self.must_not_neural_tone_words = {
            '男子', '女子', '分子', '原子', '量子', '蓮子', '石子', '瓜子', '電子', '人人', '虎虎',
            '幺幺', '幹嘛', '學子', '哈哈', '數術', '裊裊', '局地', '以下', '娃哈哈', '花花草草', '留得',
            '耕地', '想想', '熙熙', '攘攘', '卵子', '死死', '冉冉', '恳恳', '佼佼', '吵吵', '打打',
            '考考', '整整', '莘莘', '落地', '算子', '家家戶戶'
        }
        self.punc = "：，；。？！“”‘’':,;.?!"

    # the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
    # e.g.
    # word: "家里"
    # pos: "s"
    # finals: ['ia1', 'i3']
    def _neural_sandhi(self, word: str, pos: str,
                       finals: List[str]) -> List[str]:
        if word in self.must_not_neural_tone_words:
            return finals
        # reduplication words for Na e.g. 奶奶
        for j, item in enumerate(word):
            if j - 1 >= 0 and item == word[j - 1] and pos in ["Na"]:
                finals[j] = finals[j][:-1] + "5"
        
        ge_idx = word.find("個")
        if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛阿哦耶":
            finals[-1] = finals[-1][:-1] + "5"
        elif len(word) >= 1 and word[-1] in "的":
            finals[-1] = finals[-1][:-1] + "5"
        # e.g. 走了, 看着
        elif len(word) == 1 and word in "了著" and pos in {"Di"}:
            finals[-1] = finals[-1][:-1] + "5"
        elif len(word) > 1 and word[-1] in "們子" and pos in {"Nh", "Na"}:
            finals[-1] = finals[-1][:-1] + "5"
        # 个做量词
        elif (ge_idx >= 1 and
              (word[ge_idx - 1].isnumeric() or
               word[ge_idx - 1] in "幾有两半多各整每做是")) or word == '個':
            finals[ge_idx] = finals[ge_idx][:-1] + "5"

        word_list = self._split_word(word)
        finals_list = [finals[:len(word_list[0])], finals[len(word_list[0]):]]

        finals = sum(finals_list, [])
        return finals

    def _bu_sandhi(self, word: str, finals: List[str]) -> List[str]:
        for i, char in enumerate(word):
            # "不" before tone4 should be bu2, e.g. 不怕
            if char == "不" and i + 1 < len(word) and finals[i + 1][-1] == "4":
                finals[i] = finals[i][:-1] + "2"
        return finals

    def _yi_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # "一" in number sequences, e.g. 一零零, 二一零
        if word.find("一") != -1 and all(
            [item.isnumeric() for item in word if item != "一"]):
            return finals
        # "一" between reduplication words shold be yi5, e.g. 看一看
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            if word[0] in '看':
                finals[1] = finals[1][:-1] + "2"
            else:
                finals[1] = finals[1][:-1] + "4"
        # when "一" is ordinal word, it should be yi1
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for i, char in enumerate(word):
                if char == "一" and i + 1 < len(word):
                    # "一" before tone4 should be yi2, e.g. 一段
                    if finals[i + 1][-1] == '4':
                        finals[i] = finals[i][:-1] + "2"
                    # "一" before non-tone4 should be yi4, e.g. 一天
                    else:
                        # "一" 后面如果是标点，还读一声
                        if word[i + 1] not in self.punc:
                            finals[i] = finals[i][:-1] + "4"
        return finals

    def _split_word(self, word: str) -> List[str]:
        word_list = jieba.cut_for_search(word)
        word_list = sorted(word_list, key=lambda i: len(i), reverse=False)
        first_subword = word_list[0]
        first_begin_idx = word.find(first_subword)
        if first_begin_idx == 0:
            second_subword = word[len(first_subword):]
            new_word_list = [first_subword, second_subword]
        else:
            second_subword = word[:-len(first_subword)]
            new_word_list = [second_subword, first_subword]
        return new_word_list

    def _three_sandhi(self, word: str, finals: List[str]) -> List[str]:

        if len(word) == 2 and self._all_tone_three(finals):
            finals[0] = finals[0][:-1] + "2"
        elif len(word) == 3:
            word_list = self._split_word(word)
            if self._all_tone_three(finals):
                #  disyllabic + monosyllabic, e.g. 蒙古/包
                if len(word_list[0]) == 2:
                    finals[0] = finals[0][:-1] + "2"
                    finals[1] = finals[1][:-1] + "2"
                #  monosyllabic + disyllabic, e.g. 纸/老虎
                elif len(word_list[0]) == 1:
                    finals[1] = finals[1][:-1] + "2"
            else:
                finals_list = [
                    finals[:len(word_list[0])], finals[len(word_list[0]):]
                ]
                if len(finals_list) == 2:
                    for i, sub in enumerate(finals_list):
                        # e.g. 所有/人
                        if self._all_tone_three(sub) and len(sub) == 2:
                            finals_list[i][0] = finals_list[i][0][:-1] + "2"
                        # e.g. 好/喜欢
                        elif i == 1 and not self._all_tone_three(sub) and finals_list[i][0][-1] == "3" and \
                                finals_list[0][-1][-1] == "3":

                            finals_list[0][-1] = finals_list[0][-1][:-1] + "2"
                        finals = sum(finals_list, [])
        # split idiom into two words who's length is 2
        elif len(word) == 4:
            finals_list = [finals[:2], finals[2:]]
            finals = []
            for sub in finals_list:
                if self._all_tone_three(sub):
                    sub[0] = sub[0][:-1] + "2"
                finals += sub

        return finals

    def _all_tone_three(self, finals: List[str]) -> bool:
        return all(x[-1] == "3" for x in finals)

    # merge "不" and the word behind it
    # if don't merge, "不" sometimes appears alone, which may occur sandhi error
    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        last_word = ""
        for word, pos in seg:
            if last_word == "不":
                word = last_word + word
            if word != "不":
                new_seg.append((word, pos))
            last_word = word[:]
        if last_word == "不":
            new_seg.append((last_word, 'D'))
            last_word = ""
        return new_seg

    # function 1: merge "一" and reduplication words in it's left and right, e.g. "聽","一","聽" ->"聽一聽"
    # function 2: merge single  "一" and the word behind it
    # if don't merge, "一" sometimes appears alone according to jieba, which may occur sandhi error
    # e.g.
    # input seg: [('聽', 'VE'), ('一', 'D'), ('聽', 'VE')]
    # output seg: [['聽一聽', 'v']]
    def _merge_yi(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        # function 1
        for i, (word, pos) in enumerate(seg):
            if i - 1 >= 0 and word == "一" and i + 1 < len(seg) and seg[i - 1][
                    0] == seg[i + 1][0] and seg[i - 1][1] == "VE":
                if i - 1 < len(new_seg):
                    new_seg[i -
                            1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]
                else:
                    new_seg.append([word, pos])
                    new_seg.append([seg[i + 1][0], pos])
            else:
                if i - 2 >= 0 and seg[i - 1][0] == "一" and seg[i - 2][
                        0] == word and pos == "VE":
                    continue
                else:
                    new_seg.append([word, pos])
        seg = new_seg
        new_seg = []
        # function 2
        for i, (word, pos) in enumerate(seg):
            if new_seg and new_seg[-1][0] == "一":
                new_seg[-1][0] = new_seg[-1][0] + word
            else:
                new_seg.append([word, pos])
        return new_seg

    # the first and the second words are all_tone_three
    def _merge_continuous_three_tones(
            self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        sub_finals_list = [
            lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            if i - 1 >= 0 and self._all_tone_three(
                    sub_finals_list[i - 1]) and self._all_tone_three(
                        sub_finals_list[i]) and not merge_last[i - 1]:
                # if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
                if not self._is_reduplication(seg[i - 1][0]) and len(
                        seg[i - 1][0]) + len(seg[i][0]) <= 3:
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    new_seg.append([word, pos])
            else:
                new_seg.append([word, pos])

        return new_seg

    def _is_reduplication(self, word: str) -> bool:
        return len(word) == 2 and word[0] == word[1]

    # the last char of first word and the first char of second word is tone_three
    def _merge_continuous_three_tones_2(
            self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        sub_finals_list = [
            lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for (word, pos) in seg
        ]
        assert len(sub_finals_list) == len(seg)
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            if i - 1 >= 0 and sub_finals_list[i - 1][-1][-1] == "3" and sub_finals_list[i][0][-1] == "3" and not \
                    merge_last[i - 1]:
                # if the last word is reduplication, not merge, because reduplication need to be _neural_sandhi
                if not self._is_reduplication(seg[i - 1][0]) and len(
                        seg[i - 1][0]) + len(seg[i][0]) <= 3:
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    new_seg.append([word, pos])
            else:
                new_seg.append([word, pos])
        return new_seg

    def _merge_reduplication(
            self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        new_seg = []
        for i, (word, pos) in enumerate(seg):
            if new_seg and word == new_seg[-1][0]:
                new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
            else:
                new_seg.append([word, pos])
        return new_seg

    def pre_merge_for_modify(
            self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seg = self._merge_bu(seg)
        seg = self._merge_yi(seg)
        seg = self._merge_reduplication(seg)
        seg = self._merge_continuous_three_tones(seg)
        seg = self._merge_continuous_three_tones_2(seg)
        return seg

    def modified_tone(self, word: str, pos: str,
                      finals: List[str]) -> List[str]:

        finals = self._bu_sandhi(word, finals)
        finals = self._yi_sandhi(word, finals)
        finals = self._neural_sandhi(word, pos, finals)
        finals = self._three_sandhi(word, finals)
        return finals
