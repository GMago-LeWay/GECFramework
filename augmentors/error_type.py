import json
import logging
from tqdm import tqdm

import jieba
from pypinyin import pinyin, Style
from opencc import OpenCC
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader, random_split

from utils import *

logger = logging.getLogger(__name__)

class ErrorTypeFZ:
    def __init__(self) -> None:
        self.ciyu_dict = {}
        with open("fangzheng_v2/data/word_freq_dict.txt", 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                for k, v in data.items():
                    if v < 500:
                        continue
                    self.ciyu_dict[k] = v
        # 易混淆词
        self.yihunxiao_dic = {}
        with open("fangzheng_v2/data/易混淆词-补充_2.txt", encoding="utf-8") as f:
            text = f.read()
            confuse_pairs = text.split("\n\t\n")
            for temp_pair in confuse_pairs:
                pairs = []
                for line in temp_pair.split("\n"):
                    word = line.strip().split("\t")[0]
                    pairs.append(word)
                for i in range(len(pairs)):
                    self.yihunxiao_dic[pairs[i]] = []
                    for j in range(len(pairs)):
                        if i == j:
                            continue
                        self.yihunxiao_dic[pairs[i]].append(pairs[j])

        # 术语
        self.shuyu_dict = {}
        with open("fangzheng_v2/data/术语.txt", "r", encoding="utf-8") as f:
            for line in f:
                self.shuyu_dict[line.strip()] = 0

        # 成语
        self.chengyu_dict = {}
        with open("fangzheng_v2/data/成语词典_1.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                self.chengyu_dict[line] = 0

        # 异体词
        self.yitizi_dict = {}
        with open("fangzheng_v2/data/异体字.txt", "r", encoding="utf-8") as f:
            for line in f:
                l = line.replace("\n", "").split("\t")
                self.yitizi_dict[l[1]] = l[0]

        # 非推荐词
        self.feituijianci_dict = {}
        with open("fangzheng_v2/data/推荐词与非推荐词0818.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                lines = line.replace("\n", "").split("\t")
                self.feituijianci_dict[lines[1]] = lines[0]

        # 易混淆词
        self.yihunxiao_dic = {}
        with open("fangzheng_v2/data/易混淆词-补充_2.txt", encoding="utf-8") as f:
            text = f.read()
            confuse_pairs = text.split("\n\t\n")
            for temp_pair in confuse_pairs:
                pairs = []
                for line in temp_pair.split("\n"):
                    word = line.strip().split("\t")[0]
                    pairs.append(word)
                for i in range(len(pairs)):
                    self.yihunxiao_dic[pairs[i]] = []
                    for j in range(len(pairs)):
                        if i == j:
                            continue
                        self.yihunxiao_dic[pairs[i]].append(pairs[j])

        # 连词
        self.lianci_dict = {}
        with open("fangzheng_v2/data/连词0616_1.txt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                self.lianci_dict[line] = 0

        # 代词
        # daici_list=['你', '我', '他', '它', '这', '那']
        self.daici_dict = {}
        with open("fangzheng_v2/data/代词.txt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                self.daici_dict[line] = 0

        # 助词
        # zhuci_list=['啊', '么', '吗', '嘛']
        self.zhuci_dict = {}
        with open("fangzheng_v2/data/助词.txt", "r", encoding="utf-8") as f:
            for line in f:
                self.zhuci_dict[line.strip()] = 0

        # 介词
        self.jieci_dict = {}
        with open("fangzheng_v2/data/介词.txt", "r", encoding="utf-8") as f:
            for line in f:
                self.jieci_dict[line.strip()] = 0

        # 量词
        self.pattern_dic = {}
        self.pre_chars = {}
        with open("fangzheng_v2/data/pre_chars_1.txt", encoding="utf-8") as f:
            for line in f:
                self.pre_chars[line.replace("\n", "")] = 1
        quant_chars = {}
        with open("fangzheng_v2/data/quant_chars_1.txt", encoding="utf-8") as f:
            for line in f:
                quant_chars[line.replace("\n", "")] = 1
        for pre_ch in self.pre_chars:
            for quant_ch in quant_chars:
                self.pattern_dic[pre_ch + quant_ch] = 1

        self.fangweici_dict = {}
        with open("fangzheng_v2/data/方位词.txt", encoding="utf-8") as f:
            for line in f:
                self.fangweici_dict[line.strip()] = 0

        # 同音成词
        self.tongyin_words_dic = {}
        # with open(r"data/现汉+人民_同音音近词典_过滤易混淆词100.txt", encoding="utf-8") as f:
        #     for line in f:
        #         l = line.replace("\n", "").split("\t")
        #         tongyin_words_dic[l[0]] = l[1:]
        with open("fangzheng_v2/data/tongyin_dict_220705.txt", encoding="utf-8") as f:
            for line in f:
                lls = line.replace("\n", "").split("\t")
                for l1 in lls[1:]:
                    if l1 not in self.tongyin_words_dic:
                        self.tongyin_words_dic[l1] = {}
                    for l2 in lls[1:]:
                        if l1 == l2:
                            continue
                        self.tongyin_words_dic[l1][l2] = 0

        # 音近成词
        self.yinjin_words_dict = {}
        # with open(r"data/现汉+人民_同音音近词典_过滤易混淆词100.txt", encoding="utf-8") as f:
        #     for line in f:
        #         l = line.replace("\n", "").split("\t")
        #         tongyin_words_dic[l[0]] = l[1:]
        with open("fangzheng_v2/data/yinjin_dict_220704.txt", encoding="utf-8") as f:
            for line in f:
                lls = line.replace("\n", "").split("\t")
                for l1 in lls[1:]:
                    if l1 not in self.yinjin_words_dict:
                        self.yinjin_words_dict[l1] = {}
                    for l2 in lls[1:]:
                        if l1 == l2:
                            continue
                        self.yinjin_words_dict[l1][l2] = 0

        # 同音
        self.tongyin_dict = dict()
        with open("fangzheng_v2/data/同音字典220707.txt", 'r', encoding="utf-8") as f3:
            for line in f3:
                lines = line.replace("\n", "").split("\t")
                self.tongyin_dict[lines[0]] = {char: 0 for char in lines[1]}

        # 形近
        self.xingjin_dict = dict()
        with open("fangzheng_v2/data/形近混淆集all2.txt", 'r', encoding="utf-8") as f4:
            for line in f4:
                lines = line.replace("\n", '').split("\t")
                for i in range(len(lines)):
                    self.xingjin_dict[lines[i]] = "".join(lines)

        self.dict_yinjin = {"z": "zh", "zh": "z", "c": "ch", "ch": "c", "s": "sh", "sh": "s",
                        "an": "ang", "ang": "an", "en": "eng", "eng": "en", "in": "ing", "ing": "in", "un": "ong",
                        "ong": "un", "n": "l", "l": "n", "f": "h", "h": "f"}
        self.dict_yunmu = {"a":0, "o":0, "e":0, "i":0, "u":0, "v":0,
                    "ai":0, "ei":0, 'ui':0, "ao":0, "ou":0, "iu":0, "ie":0, "ue":0, "er":0,
                    "an":0, "en":0, "in":0, "un":0, "vn":0, "ang":0, "eng":0, "ing":0, "ong":0}

    def type_spelling_check_task(self, src, tgt):
        """
        Get Error Type of Spelling Check Task. Feature: equal sentence length.
        """
        assert len(src) == len(tgt)
        rig_seg = jieba.lcut(tgt)
        types = {}
        for i in range(len(src)):
            if src[i] != tgt[i]:
                ori_word, rig_word = ErrorTypeFZ.get_word2(rig_seg, i, src, tgt)
                eType = self.get_error_type(ori_word, rig_word, i, tgt)
                if eType not in types:
                    types[eType] = 1
                else:
                    types[eType] += 1
        return types


    @staticmethod
    def get_word2(ori_seg, char_index, lines0, lines1):
        temp_ind = 0
        ori_start = char_index
        ori_end = char_index + 1
        for i, seg in enumerate(ori_seg):
            temp_ind += len(seg)
            if char_index < temp_ind:
                ori_end = temp_ind
                ori_start = temp_ind - len(seg)
                break
        lines0_word = lines0[ori_start: ori_end]
        lines1_word = lines1[ori_start: ori_end]

        return lines0_word, lines1_word

    def get_shuyu(self, text):
        shuyu_id = {}
        for key in self.shuyu_dict:
            if key in text:
                index = 0
                while True:
                    index = text.find(key, index, len(text))
                    if index == -1:
                        break
                    else:
                        for i in range(len(key)):
                            shuyu_id[index+i] = 0
                        index += 1
        return shuyu_id

    def get_chengyu(self, text):
        chengyu_id = {}
        for key in self.chengyu_dict:
            if key in text:
                index = 0
                while True:
                    index = text.find(key, index, len(text))
                    if index == -1:
                        break
                    else:
                        for i in range(len(key)):
                            chengyu_id[index + i] = 0
                        index += 1
        return chengyu_id

    # 做作
    def zuozuo(self, word1, word2):
        flag = True
        for char1, char2 in zip(word1, word2):
            if char1 == char2:
                continue
            if char1 not in ["做", "作"] or char2 not in ["做", "作"]:
                flag = False
        return flag

    def dedide(self, content, lookup):
        flag = True
        for char1, char2 in zip(content, lookup):
            if char1 != char2 and char1 in ["的", "地", "得"] and char2 in ["的", "地", "得"]:
                pass
            elif char1 != char2:
                flag = False
        return flag

    def yitizi(self, word1, word2):
        flag = True
        for char1, char2 in zip(word1, word2):
            if char1 == char2:
                continue
            if char1 in self.yitizi_dict and char2 == self.yitizi_dict[char1]:
                pass
            else:
                flag = False
        return flag

    # 繁体字
    def traditional2simple(self, char):
        """繁体转简体"""

        cc = OpenCC('t2s')
        # print(cc.convert(line))

        return cc.convert(char)


    def fantizi(self, word1, word2):
        flag = True
        for char1, char2 in zip(word1, word2):
            if char1 == char2:
                continue
            jian_char = self.traditional2simple(char1)
            if jian_char == char2:
                pass
            else:
                flag = False
        return flag

    def feituijianci(self, word1, word2):
        if word2 in self.feituijianci_dict and word1 in self.feituijianci_dict[word2]:
            return True
        else:
            return False

    def yihunxiao(self, word1, word2):
        if word1 in self.yihunxiao_dic and word2 in self.yihunxiao_dic[word1]:
            return True
        flag = True
        for char1, char2 in zip(word1, word2):
            if char1 == char2:
                continue
            if char1 in self.yihunxiao_dic and char2 in self.yihunxiao_dic[char1]:
                pass
            else:
                flag = False
        return flag

    def liangci(self, word1, word2, i, lines1):
        start = 0
        # while True:
        #
        #     if index == -1:
        #         break
        #     if index <= i < index+len(word2):
        #         break
        #     start = index
        #
        if i - len(word2) + 1 > 0:
            start = i - len(word2) + 1
        index = lines1.find(word2, start, len(lines1))
        cha = i - index
        #print(cha, i, index, start, lines1, word2, word1)
        if i>0 and lines1[i-1] + word2[cha] in self.pattern_dic and lines1[i-1] + word1[cha] in self.pattern_dic:
            return True
        if i>1 and lines1[i-1] == word2[cha] and lines1[i-2] + word1[cha] in self.pattern_dic and lines1[i-2] + word1[cha] in self.pattern_dic:
            return True

        return False

    def lianci(self, word1, word2):
        if word1 in self.lianci_dict and word2 in self.lianci_dict:
            return True
        else:
            return False

    def daici(self, word1, word2):
        if word1 in self.daici_dict and word2 in self.daici_dict:
            return True
        else:
            for key in self.daici_dict:
                if key in word1:
                    index = word1.find(key)
                    if word2[index: index+len(key)] in self.daici_dict and key != word2[index: index+len(key)]:
                        return True
        return False

    def zhuci(self, word1, word2):
        if word1 in self.zhuci_dict and word2 in self.zhuci_dict:
            return True

        """ 
        flag = True
        for char1, char2 in zip(word1, word2):
            if char1 == char2:
                continue
            if char2 not in zhuci_dict or char1 not in zhuci_dict:
                flag = False
        return flag
        """
        return False

    def jieci(self, word1, word2):
        if word1 in self.jieci_dict and word2 in self.jieci_dict:
            return True
        else:
            return False

    def fangweici(self, word1, word2):
        if word1 in self.fangweici_dict and word2 in self.fangweici_dict:
            return True
        else:
            return False

    def tongyin_words(self, word1, word2):
        # print(word1, word2)
        if word1 in self.tongyin_words_dic and word2 in self.tongyin_words_dic[word1]:
            return True
        else:
            return False

    def tongyin(self, content, lookup):
        flag = True
        for char1, char2 in zip(content, lookup):
            if char1 != char2 and char1 in self.tongyin_dict and char2 in self.tongyin_dict[char1]:
                pass
            elif char1 != char2:
                flag = False
        return flag

    def xingjin(self, content, lookup):
        flag = True
        for char1, char2 in zip(content, lookup):
            if char1 == char2:
                continue
            if char1 in self.xingjin_dict and char2 in self.xingjin_dict[char1]:
                pass
            else:
                #print(char1, xingjin_dict[char1])
                flag = False
        return flag

    def xingjin_word(self, char1, char2):
        if char1 in self.xingjin_dict and char2 in self.xingjin_dict[char1]:
            return True
        else:
            return False


    def yinjin_words(self, word1, word2):
        if word1 in self.yinjin_words_dict and word2 in self.yinjin_words_dict[word1]:
            return True
        else:
            return False

    def chaifen_yunmu(self, yunmu):
        for i in range(1,len(yunmu)):
            if yunmu[:i] in self.dict_yunmu and yunmu[i:] in self.dict_yunmu:
                return [[yunmu[:i]], [yunmu[i:]]]
        # print("韵母拆分错误！{}".format(yunmu))
        return [[yunmu]]

    def get_shengmu_yunmu(self, wpinyin):
        shengmu = []
        yunmu = []
        for wpy in wpinyin:
            if "zh" in wpy[0] or "ch" in wpy[0] or "sh" in wpy[0]:
                shengmu.append([wpy[0][:2]])
                if wpy[0][2:] in self.dict_yunmu:
                    yunmu.append([wpy[0][2:]])
                else:
                    if len(wpy[0][2:]) == 0:
                        yunmu.append([[""]])
                    else:
                        if "00" in wpy[0][2:]:
                            print(wpy)
                        yunmu += self.chaifen_yunmu(wpy[0][2:])
            elif wpy[0] in self.dict_yunmu:
                shengmu.append([''])
                yunmu.append(wpy)
            else:
                shengmu.append([wpy[0][0]])
                if wpy[0][1:] in self.dict_yunmu:
                    yunmu.append([wpy[0][1:]])
                else:
                    if len(wpy[0][2:]) == 0:
                        yunmu.append([""])
                    else:
                        if "00" in wpy[0][1:]:
                            print(wpy)
                        yunmu += self.chaifen_yunmu(wpy[0][1:])

        return shengmu, yunmu

    def yinjin(self, content, lookup):
        content_pinyin = pinyin(content, style=Style.NORMAL)
        content_shengmu, content_yunmu = self.get_shengmu_yunmu(content_pinyin)
        lookup_pinyin = pinyin(lookup, style=Style.NORMAL)
        lookup_shengmu, lookup_yunmu = self.get_shengmu_yunmu(lookup_pinyin)
        #print(content_pinyin, lookup_pinyin)
        #print(content_shengmu, content_yunmu, lookup_shengmu, lookup_yunmu)
        k1 = 0  # 记录有多少个字不同
        k2 = 0  # 记录有几个相似的音
        flag_s = False
        # flag_y = False
        if len(content_shengmu) != len(lookup_shengmu) or len(content_pinyin) != len(lookup_pinyin):
            return False
        for c, l in zip(content_pinyin, lookup_pinyin):
            if c[0] != l[0]:
                k1 += 1
        if k1 == 0:
            return False

        for c_py, l_py in zip(content_shengmu + content_yunmu, lookup_shengmu + lookup_yunmu):
            if c_py[0] == l_py[0]:
                continue
            if c_py[0] in self.dict_yinjin and l_py[0] == self.dict_yinjin[c_py[0]]:
                flag_s = True
                k2 += 1
            else:
                flag_s = False
                break
        if k1 != k2:
            return False
        # if (flag_s and not flag_y) or (not flag_s and flag_y):
        if flag_s:
            return True
        else:
            return False

    def zifushu(self, word1, word2):
        k = 0
        for char1, char2 in zip(word1, word2):
            if char1 != char2:
                k += 1
        return k

    def levenshtein_distance(self, str1, str2):
        """
        计算字符串 str1 和 str2 的编辑距离
        :param str1
        :param str2
        :return:编辑距离矩阵
        """
        matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    d = 0
                else:
                    d = 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
                # print(matrix[i])
        return matrix



    def levenshtein_distinguish(self, word1, word2):
        """
        比较字符串 str1 和 str2 的不一致
        :param str1：原句子
        :param str2：新句子
        :return: 结果list
        """
        #print(word1)
        #print(word2)
        dp = self.levenshtein_distance(word1, word2)
        m = len(dp) - 1
        n = len(dp[0]) - 1
        spokenstr = []
        writtenstr = []
        while n >= 0 or m >= 0:
            if n and dp[m][n - 1] + 1 == dp[m][n]:

                spokenstr.append("insert")
                writtenstr.append(word2[n - 1])
                n -= 1
                continue
            if m and dp[m - 1][n] + 1 == dp[m][n]:
                # print("delete %c" %(word1[m-1]))

                spokenstr.append(word1[m - 1])
                writtenstr.append("delete")


                m -= 1
                continue
            if dp[m - 1][n - 1] + 1 == dp[m][n]:
                # print("replace %c %c" %(word1[m-1],word2[n-1]))

                spokenstr.append(word1[m - 1])
                writtenstr.append(word2[n - 1])

                n -= 1
                m -= 1
                continue
            if dp[m - 1][n - 1] == dp[m][n]:
                spokenstr.append(' ')
                writtenstr.append(' ')
            n -= 1
            m -= 1
        spokenstr = spokenstr[::-1]
        writtenstr = writtenstr[::-1]
        edit_dis = 0
        for temp in spokenstr:
            if temp != " ":
                edit_dis += 1
        return edit_dis

    def get_diff_num(self, ori_piyin, pre_piyin):
        num = 0
        for i in range(len(ori_piyin)):
            if ori_piyin[i] != pre_piyin[i]:
                num += 1
        return num
        
    # 相似拼音
    def pinyin_smi_word(self, word1, word2):
        flag = True
        for char1, char2 in zip(word1, word2):
            if char1 == char2:
                continue
            pinyin1 = pinyin(char1, style=Style.NORMAL)[0][0]
            pinyin2 = pinyin(char2, style=Style.NORMAL)[0][0]
            if pinyin1 == pinyin2:
                continue
            if self.levenshtein_distinguish(pinyin1, pinyin2) == 1:
                pass
            elif len(pinyin1) == len(pinyin2) and self.levenshtein_distinguish(pinyin1, pinyin2) == 2 and self.get_diff_num(pinyin1, pinyin2) == 0:
                pass
            else:
                flag = False
        return flag

    def get_error_type(self, ori_word, pre_word, i, lines1):
        if ori_word in self.ciyu_dict:
            ori_chengci = True
        else:
            ori_chengci = False
        if pre_word in self.ciyu_dict:
            pre_chengci = True
        else:
            pre_chengci = False
        if len(ori_word) != len(pre_word):
            print("{} 与 {} 长度不一致".format(ori_word, pre_word))
        entity_id = {}
        shuyu_id = self.get_shuyu(lines1)
        chengyu_id = self.get_chengyu(lines1)
        if i in entity_id:
            error_tpye = "实体"
        elif i in chengyu_id:
            error_tpye = "成语"
        elif self.zuozuo(ori_word, pre_word):
            error_tpye = "做-作"
        elif self.dedide(ori_word, pre_word):
            error_tpye = "的地得"
        elif self.yitizi(ori_word, pre_word):
            error_tpye = "异体字"
        elif self.fantizi(ori_word, pre_word):
            error_tpye = "繁体字"
        elif self.feituijianci(ori_word, pre_word):
            error_tpye = "非推荐词"
        elif self.yihunxiao(ori_word, pre_word):
            error_tpye = "易混淆"
        elif i in shuyu_id:
            error_tpye = "术语"
        elif i > 0 and self.liangci(ori_word, pre_word, i, lines1):
            error_tpye = "量词"
        elif self.lianci(ori_word, pre_word):
            error_tpye = "连词"
        elif self.daici(ori_word, pre_word):
            error_tpye = "代词"
        elif self.zhuci(ori_word, pre_word):
            error_tpye = "助词"
            # 都在词典里
        elif self.jieci(ori_word, pre_word):
            error_tpye = "介词"
            # 都在词典里
        elif self.fangweici(ori_word, pre_word):
            error_tpye = '方位词'
        elif len(ori_word) > 1 and self.tongyin_words(ori_word, pre_word):
            # print(ori_word, pre_word)
            error_tpye = "同音-前后都在词典"
        elif len(ori_word) > 1 and ori_chengci and pre_chengci and self.xingjin_word(ori_word, pre_word):
            error_tpye = "形近-前后都在词典"
        elif len(ori_word) > 1 and self.yinjin_words(ori_word, pre_word):
            error_tpye = "音近-前后都在词典"
        elif self.tongyin(ori_word, pre_word):
            # print(ori_word, pre_word)
            if pre_chengci:
                error_tpye = "同音-改后在词典"
            elif len(ori_word) > 1:
                error_tpye = '同音-改后在结巴词典'
            else:
                error_tpye = "同音-其他"
        elif self.xingjin(ori_word, pre_word):
            if pre_chengci:
                error_tpye = "形近-改后在词典"
            elif len(ori_word) > 1:
                error_tpye = '形近-改后在结巴词典'
            else:
                error_tpye = "形近-其他"
        elif self.yinjin(ori_word, pre_word):
            if pre_chengci:
                error_tpye = "音近-改后在词典"
            elif len(ori_word) > 1:
                error_tpye = '音近-改后在结巴词典'
            else:
                error_tpye = "音近-其他"
        elif ori_chengci and pre_chengci:
            # print(ori_word, pre_word)
            if self.zifushu(ori_word, pre_word) > 1:
                error_tpye = "非同音形近音近-前后都在词典-多错误"
            else:
                error_tpye = "非同音形近音近-前后都在词典-单错误"
        elif self.pinyin_smi_word(ori_word, pre_word):
            if pre_chengci:
                error_tpye = "拼音相似-改后在词典"
            elif len(ori_word) > 1:
                error_tpye = '拼音相似-改后在结巴词典'
            else:
                error_tpye = "拼音相似-其他"
        elif pre_chengci:
            error_tpye = "非同音形近音近-改后在词典"

        elif len(ori_word) > 1:
            error_tpye = "非同音形近音近-改后在结巴词典"
        elif len(ori_word) == 1:
            error_tpye = "非同音形近音近-其他"
        else:
            error_tpye = "其他"
        return error_tpye



class BertForAugmentation:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

        assert self.args.model in ['bert'], "check model in args, current augmentation requires bert."
        # bert model
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.model = AutoModelForMaskedLM.from_pretrained(config.pretrained_model).to(args.device)
        self.mask_id = self.tokenizer.mask_token_id


    def _get_collate_fn(self):

        def collate_fn(batch):
            batch_size = len(batch)
            masked_text = [batch[i][1] for i in range(batch_size)]
            correct_character = [batch[i][0][batch[i][2]] for i in range(batch_size)]

            masked_text = self.tokenizer(masked_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            correct_character = self.tokenizer(correct_character, padding=True, truncation=True,  max_length=512, return_tensors="pt")
            pos = (masked_text['input_ids'] == self.mask_id).long()
            assert pos.sum().item() == batch_size, "Not single error character in one sentence"

            return {"masked_text": masked_text, "pos": pos, "correct_character": correct_character["input_ids"][:, 1]}

        return collate_fn


    def inference(self, dataloader: DataLoader):
        self.model.eval()

        # results save
        wrong_characters: list[str] = []
        augment_flags: list[bool] = []

        topk = 10
        for epoch, batch in tqdm(enumerate(dataloader)):
            masked_text = batch["masked_text"].to(self.args.device)
            pos = batch["pos"].to(self.args.device)
            batch_size = pos.size(0)
            correct_character_id = batch["correct_character"]

            logits = self.model(**masked_text).logits
            single_logits = (logits * pos.unsqueeze(-1)).sum(dim=1).detach().cpu()
            topk_logits, topk_id = single_logits.topk(topk)
            topk_logits, topk_id = topk_logits.detach().cpu(), topk_id.detach().cpu()

            # find the max rank char except right char
            for i in range(batch_size):
                find = False
                for j in range(topk):
                    if topk_id[i][j] != correct_character_id[i]:
                        decode_char = self.tokenizer.decode(topk_id[i][j])
                        if len(decode_char) == 1 and is_chinese(decode_char):
                            wrong_characters.append(decode_char)
                            augment_flags.append(True)
                            find = True
                            break
                if find == False:
                    logger.info("Warning: did not find suitable substitution")
                    wrong_characters.append(self.tokenizer.decode(correct_character_id[i]))
                    augment_flags.append(False)


        return wrong_characters, augment_flags


    def augment(self, data_list:list[str], mask_position: list[int]) -> list[str]:
        data = [] # [src_text, masked_text, mask_position]
        for i in range(len(data_list)):
            masked_text = data_list[i][:mask_position[i]] + '[MASK]' + data_list[i][mask_position[i]+1:]
            data.append([data_list[i], masked_text, mask_position[i]])

        dataloader = DataLoader(data, batch_size=self.config.batch_size, collate_fn=self._get_collate_fn(), drop_last=False, shuffle=False)
        wrong_characters, augment_flags = self.inference(dataloader)
        assert len(wrong_characters) == len(data_list)

        result = []
        for i in range(len(data_list)):
            augment_text = data[i][1].replace('[MASK]', wrong_characters[i])
            result.append(augment_text)
        return result, augment_flags


