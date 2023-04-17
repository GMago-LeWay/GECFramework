from typing import List, Union, Optional
import re
import random
import copy
import numpy as np
import json
import logging
from tqdm import tqdm
from functools import wraps, lru_cache

import synonyms
import time

logger = logging.getLogger(__name__)

def get_time():
    localTime = time.localtime(time.time()) 
    strTime = '[' + time.strftime("%Y-%m-%d %H:%M:%S", localTime) + '] ' 
    return strTime
########## segmentor.py ###############

# n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名
# m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词
# v/动词 a/形容词 d/副词 h/前接成分 k/后接成分
# i/习语 j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词
# e/叹词 o/拟声词 g/语素 w/标点 x/其它
segmentor = None

@lru_cache(maxsize=1000)
def word_seg(sent):
    global segmentor
    if segmentor is None:
        import thulac
        segmentor = thulac.thulac()
    return segmentor.cut(sent)

def seg_dec(func):
    @wraps(func)
    def _func(obj, sent, *args, **kwargs):
        if isinstance(sent, str):
            original_sent = sent
            sent = word_seg(sent)
        else:
            original_sent = "".join(map(lambda x: x[0], sent))
        result, rules = func(obj, sent, *args, **kwargs)
        for i in range(len(result)):
            if isinstance(result[i], list):
                # ["xxx", "yyy", ...]
                if isinstance(result[i][0], str):
                    result[i] = "".join(result[i])
                # [["xxx", "n"], ["yyy", "v"], ...]
                elif isinstance(result[i][0], list):
                    result[i] = "".join(map(lambda x: x[0], result[i]))
                # else: "xxxyyy..."
        return {"type": obj.__class__.__name__, "origin": original_sent, "transform": result, "rules": rules}
    return _func

########## augmentor.py ###############

class Augmentor:
    def __init__(self):
        # 规则函数：以单下划线开头
        self.rules = [name for name in dir(self) if re.match("^_[a-zA-Z]", name) and callable(getattr(self, name))]

    @seg_dec
    def transform(self, sent: Union[list, str]) -> tuple[list, list]:
        results, rules = [], []
        for rule in self.rules:
            result = getattr(self, rule)(sent)
            if result is not None:
                results.append(result)
                rules.append(rule)
        return results, rules
    
################ improper_collocation ####################

class ImproperCollocation(Augmentor):
    def __init__(self):
        super(ImproperCollocation, self).__init__()
        self.subject_predicate_list = [
            ["目光", "集中", 0, "眼睛"], ["水平", "提高", 1, "改善"], ["条件", "改善", 1, "提高"], ["反响", "热烈", 1, "热情"],
            ["形象", "浮现", 0, "精神"], ["结果", "显示", 1, "显现"], ["成本", "增加", 1, "加强"], ["性别", "平等", 1, "相等"],
            ["基因", "表达", 1, "表现"], ["程度", "严重", 1, "严峻"], ["研究", "表明", 1, "表达"], ["研究", "发现", 1, "发掘"],
            ["技术", "发展", 1, "发现"], ["服务", "周到", 1, "严密"], ["信息", "传递", 1, "递送"], ["病毒", "传播", 1, "传达"]
        ]
        self.predicate_object_list = [
            ["加大", "力度", 0, "加强"], ["提供", "服务", 0, "供应"], ["提高", "效率", 0, "增加"], ["达成", "共识", 0, "完成"],
            ["制定", "政策", 0, "制造"], ["发表", "文章", 0, "发放"], ["采取", "行动", 0, "采纳"], ["实现", "目标", 0, "实施"],
            ["解决", "问题", 0, "决定"], ["传递", "信息", 0, "流传"], ["开展", "活动", 0, "开发"], ["取得", "成就", 0, "开创"],
            ["建立", "模型", 0, "树立"], ["发挥", "作用", 0, "发生"], ["提出", "要求", 0, "提取"], ["创造", "价值", 0, "造成"],
            ["打破", "纪录", 0, "破除"], ["预防", "疾病", 0, "提防"], ["吸取", "教训", 0, "听取"], ["面临", "挑战", 0, "面向"]
        ]
        self.attribute_center_list = [
            ["丰富", "资源", 0, "优裕"], ["重大", "意义", 0, "杰出"], ["宽阔", "", 0, "辽阔"], ["部", "电影", 0, "台"],
            ["性能", "稳定", 1, "安定"], ["密切", "联系", 0, "亲切"], ["热烈", "欢迎", 0, "激烈"], ["慎重", "考虑", 0, "庄重"],
            ["熟练", "掌握", 0, "老练"], ["沉重", "打击", 0, "繁重"], ["幸福", "生活", 0, "幸运"], ["迫切", "需要", 0, "紧迫"],
            ["恶劣", "环境", 0, "劣质"], ["密切", "合作", 0, "亲切"], ["诚信", "经营", 0, "诚实"], ["严厉", "批评", 0, "严格"]
        ]
        self.connective_list = [
            ["无论", "都", 1, "也"], ["只有", "才", 1, "就"], ["尽管", "", 0, "不管"], ["如果", "就", 1, "也"],
            ["不仅", "而且", 1, "但"], ["不仅", "还", 1, "但"], ["宁可", "也", 1, "还"], ["不管", "都", 1, "却"],
            ["虽然", "但", 1, "也"], ["与其", "不如", 1, "也不"], ["要是", "那么", 1, "也"], ["哪怕", "也", 1, "就"]
        ]

    def _replace_subject_predicate(self, sent: list) -> Optional[str]:
        return self.replace_normal_items(sent, self.subject_predicate_list)

    def _replace_predicate_object(self, sent: list) -> Optional[str]:
        return self.replace_normal_items(sent, self.predicate_object_list)

    def _replace_attribute_center(self, sent: list) -> Optional[str]:
        return self.replace_normal_items(sent, self.attribute_center_list)

    def _replace_connective(self, sent: list) -> Optional[str]:
        return self.replace_normal_items(sent, self.connective_list)

    def replace_normal_items(self, sent: list, replace_list: list) -> Optional[str]:
        text = "".join(map(lambda x: x[0], sent))
        for (word1, word2, ridx, rword) in random.sample(replace_list, k=len(replace_list)):
            search = re.search(f"{word1}.*?{word2}", text)
            if search is None:
                continue
            start, end = search.span()
            if ridx == 0:
                return text[:start] + rword + text[start + len(word1):]
            elif ridx == 1:
                return text[:end - len(word2)] + rword + text[end:]
            else:
                raise NotImplementedError
        return None


############## improper_logicality ##########################

delimiters = "，。！？：；.,!?:;"
numerals = "一二三四五六七八九0123456789"

class ImproperLogicality(Augmentor):
    def __init__(self):
        super(ImproperLogicality, self).__init__()

    def _insert_negation(self, sent: list) -> Optional[str]:
        trigger_words = ["避免", "防止", "杜绝"]
        escape_words = ["无法", "不能"]
        verbs = ["受到", "接受", "使用", "发生", "出现", "遭", "被", "遇见", "忘记", "丢失", "遗忘", "传播", "泄露"]
        text = "".join(map(lambda x: x[0], sent))
        search = re.search(f"(?<!{'|'.join(escape_words)})({'|'.join(trigger_words)})[^{delimiters}]*?({'|'.join(verbs)})", text)
        if search is not None:
            start, end = search.span()
            verb_length = len(search.group(2))
            return text[:end - verb_length] + "不" + text[end - verb_length:]
        return None

    def _insert_absolute(self, sent: list) -> Optional[str]:
        trigger_words = ["往往", "时常", "经常", "偶尔", "每每", "不时", "常常", "通常", "有时"]
        replace_words = ["总", "总是", "一定"]
        text = "".join(map(lambda x: x[0], sent))
        search = re.search(f"({'|'.join(trigger_words)})(?![{delimiters}])", text)
        if search is not None:
            select_word = random.choice(replace_words)
            trigger_word = search.group(1)
            start, end = search.span()
            return text[:start] + select_word + text[start + len(trigger_word):]
        return None

    def _replace_coordinate(self, sent: list) -> Optional[str]:
        text = "".join(map(lambda x: x[0], sent))
        candidates = ["和", "以及", "或"]
        search = re.search(f"、[^{delimiters}]+?等", text)
        if search is not None:
            start, end = search.span()
            return text[:end - 1] + random.choice(candidates) + text[end:]
        search2 = re.search(f"、[^{delimiters}]+?以及", text)
        if search2 is not None:
            start, end = search2.span()
            return text[:end - 2] + "等" + text[end:]
        return None

    def _replace_numeral(self, sent: list) -> Optional[str]:
        text = "".join(map(lambda x: x[0], sent))
        trigger_words = ["降低", "减少", "下降", "减低", "减"]
        search = re.search(f"({'|'.join(trigger_words)})(一半|[{numerals}]分之[{numerals}]|[0-9.]+%)", text)
        if search is not None:
            start, end = search.span()
            # print(search)
            scope = search.group(2)
            if scope == "一半":
                replace_word = "一倍"
            elif re.match(f"^[{numerals}]分之[{numerals}]$", scope):
                replace_word = f"{scope[0]}倍"
            elif (match := re.match(f"^([0-9.]+)%$", scope)):
                try:
                    str_num = match.group(1)
                    num = float(str_num) if "." in str_num else int(str_num)
                    while num < 100:
                        num *= 10
                    print(num)
                    replace_word = f"{num}%"
                except ValueError:
                    return None
            else:
                raise NotImplementedError
            return text[:end - len(scope)] + replace_word + text[end:]
        return None


############### improper_word_order ##########################

class ImproperWordOrder(Augmentor):
    def __init__(self):
        super(ImproperWordOrder, self).__init__()

    def _reorder_connective(self, sent: list) -> Optional[list]:
        first_connective, first_noun = -1, -1
        for i in range(len(sent)):
            if first_connective != -1 and first_noun != -1:
                break
            if sent[i][1] == "c" and first_connective == -1:
                first_connective = i
            elif sent[i][1].startswith("n") or sent[i][1] == "r" and first_noun == -1:
                first_noun = i
        if first_connective == -1 or first_noun == -1:
            return None
        output = copy.deepcopy(sent)
        # 关联词在主语前
        if first_connective < first_noun:
            noun = output.pop(first_noun)
            output.insert(first_connective, noun)
        # 关联词在主语后
        else:
            connective = output.pop(first_connective)
            output.insert(first_noun, connective)
        return output

    def _reorder_attribute_adverbial(self, sent: list) -> Optional[list]:
        positions = []
        for i in range(1, len(sent)):
            pos1, pos2 = -1, -1
            if i >= len(sent) - 1 or not (sent[i][1] == "v" and sent[i - 1][1] != "u"):
                continue
            else:
                pos1 = i
            j = i + 2 if sent[i + 1][1] == "u" else i + 1
            if j >= len(sent) - 1 or not sent[j][1] == "a":
                continue
            else:
                pos2 = j
            k = j + 2 if sent[j + 1][1] == "u" else j + 1
            if k >= len(sent) - 1 or not (sent[k][1].startswith("n") or sent[k][1] == "v"):
                continue
            positions.append((pos1, pos2))
        if positions:
            pos1, pos2 = random.choice(positions)
            output = copy.deepcopy(sent)
            item = output.pop(pos2)
            if output[pos2][1] == "u":
                output.pop(pos2)
            item[0] = item[0] + "地"
            output.insert(pos1, item)
            return output
        return None

    def _reorder_attribute_center(self, sent: list) -> Optional[list]:
        positions = []
        for i in range(len(sent) - 2):
            # (a/v, u, n) -> (n, u, a/v)
            t0, t1, t2 = sent[i:i + 3]
            if t0[1] in ["a", "v"] and t1[1] == "u" and t2[1].startswith("n"):
                positions.append(i)
        if positions:
            position = random.choice(positions)
            output = copy.deepcopy(sent)
            output[position], output[position + 2] = output[position + 2], output[position]
            return output
        return None

################# missing_component #######################

DEBUG = False

class MissingComponent(Augmentor):
    def __init__(self):
        super(MissingComponent, self).__init__()
        self.connective_list = [('不但', '还'), ('不但', '也'), ('不光', '还'), ('不仅', '而且'),
            ('不仅', '还'), ('不只', '又'),  ('不但', '并且'), ('不但', '而且'),
            ('别说', '就'), ('不单', '还'), ('不但', '反而'), ('不但', '更'),
            ('不但', '相反'), ('不光', '而且'), ('不光', '就是'), ('不光', '也'),
            ('不仅', '反而'), ('不仅', '就是'), ('不仅', '甚至'), ('不仅', '也'),
            ('不仅', '尤其'), ('不只', '还'), ('非但', '反而'), ('由于', '因此'),
            ('因为', '才'), ('因为', '便'), ('因为', '就'), ('因为', '以致'),
            ('由于', '所以'), ('即使', '也'), ('假如', '就'), ('就是', '也'),
            ('就算', '也'), ('如果', '就'), ('如果', '那么'), ('就算', '总'),
            ('既是', '也'), ('即使', '还'), ('即使', '总'), ('哪怕', '都'),
            ('倘若', '便'), ('若是', '则'), ('若是', '那'), ('若是', '就'),
            ('尽管', '可'), ('尽管', '却'), ('虽然', '可是'), ('虽然', '却'),
            ('虽然', '但'), ('虽说', '但'), ('固然', '不过'), ('固然', '但'),
            ('尽管', '也'), ('虽然', '可'), ('虽然', '然而'), ('虽说', '也'),
            ('幸而', '否则'), ('虽则', '也'), ('幸亏', '不然'), ('虽然', '还'),
            ('假使', '还'), ('要是', '那么'), ('只要', '就'), ('只有', '才'),
            ('无论', '都'), ('不管', '也'), ('因为', '所以'), ('既然', '那么'),
            ('既然', '就'), ('之所以', '是因为'), ('不是', '而是'), ('不是', '就是'),
            ('既', '又'), ('一', '就'), ('与其', '不如'), ('宁可', '也不')]
        self.verb_list = list(
            {'过着', '取得', '获得', '参加', '开展', '举办', '承担', '缺乏', '改善', '培养', '具备', '拥有', '打破',
             '打碎', '破坏', '修复', '维护', '修护', '承办', '害怕', '爱', '骂', '收拾', '推翻', '充满', '剔除', '去除',
             '取出', '祛除', '驱除', '赶出', '拿出', '袒护', '弹奏', '打开', '关闭', '欣赏', '挥舞', '挥动', '甩', '拧',
             '痛恨', '铭记', '牢记', '找', '招收', '招揽', '合上', '操作', '使用', '编辑', '手持', '批评', '宣传',
             '保卫', '研究', '打听', '聆听', '探望', '缅怀', '鄙视', '蔑视', '歧视', '打量', '遥望', '看护', '保护',
             '搀', '抱', '搂', '扶', '捉', '擒', '掐', '推', '拿', '抽', '撕', '摘', '拣', '捡', '打', '播', '击', '捏',
             '撒', '按', '弹', '撞', '提', '扭', '捶', '持', '揍', '披', '捣', '搜', '托', '举', '拖', '擦', '敲', '挖',
             '抛', '掘', '抬', '插', '扔', '写', '抄', '摇', '抓', '捧', '掷', '撑', '摊', '倒', '摔', '劈', '画', '搔',
             '撬', '挥', '揽', '挡', '捺', '抚', '搡', '拉', '摸', '拍', '剪', '拎', '拔', '拨', '舞', '握', '攥', '咬',
             '吞', '吐', '吮', '吸', '啃', '喝', '吃', '咀', '嚼', '瞥', '视', '盯', '瞧', '窥', '瞄', '眺', '瞪',
             '瞅'})
        self.attribute_list = [
            ('相当', ['a']), ('非常', ['a']), ('很', ['a']), ('极其', ['a']), ('十分', ['a']),
            ('极', ['a']), ('最', ['a']), ('顶', ['a']), ('太', ['a']), ('更', ['a']),
            ('挺', ['a']), ('格外', ['a']), ('分外', ['a']), ('更加', ['a']), ('越', ['a']),
            ('越发', ['a']), ('有点', ['a']), ('有点儿', ['a']), ('稍', ['a']), ('稍微', ['a']),
            ('稍稍', ['a']), ('略微', ['a']), ('略', ['a']), ('几乎', ['a']), ('过于', ['a']),
            ('尤其', ['a']), ('特别', ['a']), ('真', ['a']), ('真是', ['a'])
        ]
        self.adverbial_list = [
            ('分别', ['v']), ('各自', ['v']), ('各', ['v']), ('情愿', ['v']),
            ('肯', ['v']), ('要', ['v']), ('愿', ['v']), ('想要', ['v']),
            ('要想', ['v']), ('敢', ['v']), ('敢于', ['v']), ('乐于', ['v']), ('应', ['v']),
            ('应当', ['v']), ('得', ['v']), ('该', ['v']), ('当', ['v']), ('须得', ['v']),
            ('理当', ['v']), ('便于', ['v']), ('难于', ['v']), ('难以', ['v']), ('易于', ['v']),
        ]
        self.complement_list = [
            ('考虑', '一下'), ('思考', '一下'), ('想', '一下'), ('打了', '一下'), ('丢', '不得'),
            ('去', '不得'), ('大意', '不得'), ('耽误', '不得'), ('建设', '得'), ('建设', '成'),
        ]

    def _missing_connective(self, sent: list) -> Optional[str]:
        text = "".join(map(lambda x: x[0], sent))
        for (word1, word2) in random.sample(self.connective_list, k=len(self.connective_list)):
            search = re.search(f"{word1}.+?{word2}", text)
            if search is not None:
                start, end = search.span()
                return text[:end - len(word2)] + text[end:]
        return None

    def _missing_subjective(self, sent: list) -> Optional[list]:
        candidates = ["使", "使得", "令", "让"]
        for i in range(1, len(sent) - 2):
            if sent[i][0] == "，" and sent[i + 1][1] in ["np", "r"]:
                candidate = random.choice(candidates)
                return [*sent[:i + 1], [candidate, "v"], *sent[i + 1:]]
        return None

    def _missing_predicate_1(self, sent: list) -> Optional[list]:
        remove_list = []
        for i in range(1, len(sent) - 1):
            if sent[i][1] == "v" and sent[i][0] in self.verb_list:
                for j in range(i + 1, len(sent)):
                    if sent[j][1] in ["w", "x"]:
                        break
                    if sent[j][1] in ["n"]:
                        remove_list.append(i)
        if remove_list:
            ri = random.choice(remove_list)
            return sent[:ri] + sent[ri + 1:]
        return None

    def _missing_predicate_2(self, sent: list) -> Optional[list]:
        for i in range(len(sent)):
            if sent[i][1] == "v" and sent[i][0] in self.verb_list:
                return [*sent[:i], ["为", "p"], *sent[i + 1:]]
        return None

    def _missing_objective(self, sent: list) -> Optional[list]:
        remove_list = []
        for i in range(1, len(sent)):
            if sent[i][1] in ["n", "np", "r"] and sent[i - 1][1] == "v":
                remove_list.append(i)
        if remove_list:
            ri = random.choice(remove_list)
            return sent[:ri] + sent[ri + 1:]
        return None

    def _missing_attribute(self, sent: list) -> Optional[list]:
        word_list = list(map(lambda x: x[0], sent))
        for (word, pos) in random.sample(self.attribute_list, k=len(self.attribute_list)):
            if word not in word_list:
                continue
            for i in range(1, len(sent)):
                if sent[i - 1][0] == word and sent[i][1] in pos:
                    return sent[:i] + sent[i + 1:]
        return None

    def _missing_adverbial(self, sent: list) -> Optional[list]:
        word_list = list(map(lambda x: x[0], sent))
        for (word, pos) in random.sample(self.attribute_list, k=len(self.attribute_list)):
            if word not in word_list:
                continue
            for i in range(1, len(sent)):
                if sent[i - 1][0] == word and sent[i][1] in pos:
                    return sent[:i - 1] + sent[i:]
        return None

    def _missing_complement(self, sent: list) -> Optional[str]:
        text = "".join(map(lambda x: x[0], sent))
        for (word1, word2) in self.complement_list:
            word = word1 + word2
            if word in text:
                return text.replace(word, word1)
        return None

##################### redundant_component ######################

DEBUG = False

class RedundantComponent(Augmentor):
    def __init__(self, n_word_probs=(0.5, 0.3, 0.1, 0.1), k_probs=(0.15,) * 3 + (0.1,) * 4 + (0.05,) * 3):
        super(RedundantComponent, self).__init__()
        self.n_word_probs = n_word_probs
        self.k_probs = k_probs

    def _redundant_component(self, sent: Union[list, str]) -> list:
        output = []
        valid_indices = [i for i, (word, pos) in enumerate(sent) if re.match(r"[vadcp]", pos)]
        n_word = random.choices(range(1, len(self.n_word_probs) + 1), weights=self.n_word_probs)[0]
        select_indices = set(random.sample(valid_indices, k=min(n_word, len(valid_indices))))
        # print(valid_indices, n_word, select_indices)
        for i, (word, pos) in enumerate(sent):
            origin = (word, pos)
            if i in select_indices:
                rand = random.random()
                word_synonyms, _ = synonyms.nearby(word, size=len(self.k_probs) + 1)
                # print(word, word_synonyms)
                if len(word_synonyms) <= 1:
                    continue
                synonym = random.choices(word_synonyms[1:], weights=self.k_probs)[0]
                transform = (f"(I: {synonym})", pos) if DEBUG else (synonym, pos)
                if rand < 0.5:
                    output.append(transform)
                output.append(origin)
                if rand >= 0.5:
                    output.append(transform)
            else:
                output.append(origin)
        text = ""
        for item in output:
            text += item[0]
        return text
    
################### structural_confusion ##################3

delimiters = "，。！？：；.,!?:;"

class StructuralConfusion(Augmentor):
    def __init__(self):
        super(StructuralConfusion, self).__init__()
        self.mixed_patterns = [
            # 是...的结果 & 是由于... -> 是由于...的结果
            (("是", "的结果"), ("是由于", f"[{delimiters}]"), (1, 0)),
            # 以...为目的 & 是为了... -> 是为了...为目的
            (("以", "为目的"), ("是为了", f"[{delimiters}]"), (1, 0)),
            # 对于...问题 & 在...问题上 -> 对于...问题上
            (("对于", "问题"), ("在", "问题上"), (0, 1)),
            # 原因是... & 是由...造成的 -> 原因是...造成的
            (("原因是", f"[{delimiters}]"), ("是由", "造成的"), (0, 1)),
            # 靠的是... & 是靠...取得的 -> 靠的是...取得的
            (("靠的是", f"[{delimiters}]"), ("是靠", "取得的"), (0, 1)),
            # 本着...的原则 & 以...为原则 -> 本着...为原则
            (("本着", "的原则"), ("以", "为原则"), (0, 1)),
            # 成分是... & 由...配置而成 -> 成分是...配制而成
            (("成分是", f"[{delimiters}]"), ("由", "配置而成"), (0, 1)),
            # 以...为幌子 & 打着...的幌子 -> 打着...为幌子
            (("以", "为幌子"), ("打着", "的幌子"), (0, 1)),
            # 关键在于... & ...是十分重要的 -> 关键在于...是十分重要的
            (("关键在于", f"[{delimiters}]"), (f"[{delimiters}]", "是十分重要的"), (0, 1)),
            # 分为...部分 & 由...部分组成 -> 分为...部分组成
            (("分为", "部分"), ("由", "部分组成"), (0, 1)),
            # 有...部分 & 由...部分组成 -> 有...部分组成
            (("有", "部分"), ("由", "部分组成"), (0, 1)),
            # 围绕... & 以...为中心 -> 围绕...为中心
            (("围绕", f"[{delimiters}]"), ("以", "为中心"), (0, 1)),
            # 从...出发 & 以...为出发点 -> 从...为出发点
            (("从", "出发"), ("以", "为出发点"), (0, 1)),
            # 之所以... -> 之所以...的原因
            (("之所以", f"[{delimiters}]"), (f"[{delimiters}]", "的原因"), (0, 1)),
            # 听到...消息 & ...消息传来 -> 听到...消息传来
            (("听到", "消息"), (f"[{delimiters}]", "消息传来"), (0, 1)),
        ]
        self.mixed_patterns_2 = [
            (["是由于", "是因为", "原因是"], ["的原因", "的结果", "决定的", "造成的"]),
            (["高达", "长达"], ["之多", "之久"]),
        ]

    def _mix_pattern_1(self, sent: list) -> Optional[str]:
        text = "".join(map(lambda x: x[0], sent))
        for (pattern1, pattern2, (index1, index2)) in random.sample(self.mixed_patterns, k=len(self.mixed_patterns)):
            assert (index1, index2) in [(0, 1), (1, 0)]
            search = re.search(f"{pattern1[0]}[^{delimiters}]*?{pattern1[1]}", text)
            if search is not None:
                start, end = search.span()
                if index1 == 0:
                    end2 = end - 1 if text[end - 1] in delimiters else end
                    return f"{text[:end - len(pattern1[1])]}{pattern2[1]}{text[end2:]}"
                else:
                    start2 = start + 1 if text[start] in delimiters else start
                    return f"{text[:start2]}{pattern2[0]}{text[start + len(pattern1[0]):]}"
            search2 = re.search(f"{pattern2[0]}[^{delimiters}]*?{pattern2[1]}", text)
            if search2 is not None:
                start, end = search2.span()
                if index2 == 0:
                    end2 = end - 1 if text[end - 1] in delimiters else end
                    return f"{text[:end - len(pattern2[1])]}{pattern1[1]}{text[end2:]}"
                else:
                    start2 = start + 1 if text[start] in delimiters else start
                    return f"{text[:start2]}{pattern1[0]}{text[start + len(pattern2[0]):]}"
        return None

    def _mixed_pattern_2(self, sent: list) -> Optional[str]:
        text = "".join(map(lambda x: x[0], sent))
        for trigger_words, candidates in random.sample(self.mixed_patterns_2, k=len(self.mixed_patterns_2)):
            for word in trigger_words:
                search = re.search(f"{word}[^{delimiters}]*?[{delimiters}]", text)
                if search is not None:
                    start, end = search.span()
                    candidate = random.choice(candidates)
                    return f"{text[:end - 1]}{candidate}{text[end - 1:]}"
        return None

################ added* ####################

name_map = {
    "redundant_component": RedundantComponent,
    "missing_component": MissingComponent,
    "structural_confusion": StructuralConfusion,
    "improper_logicality": ImproperLogicality,
    "improper_collocation": ImproperCollocation,
    "improper_word_order": ImproperWordOrder,
}

class CLGAugment:
    def __init__(self, name) -> None:
        self.method = name_map[name]()

    def augment(self, sentence: str) -> List[dict]:
        augment_res = []
        sample = re.sub("\s+", "", sentence)
        if sample[-1] not in ['；','！','？',',','!','?','...', "。", "."]:
            sample += "。"
        output = self.method.transform(sample)
        augment_num = len(output["transform"])
        for i in range(augment_num):
            res = {"text": output["transform"][i], "label": output["origin"], "type": output["type"], "rule": output["rules"][i]}
            augment_res.append(res)
        return augment_res

class CLGAugmentor:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

        self.augmentors = []
        for name in name_map:
            self.augmentors.append(CLGAugment(name=name))

        assert len(self.augmentors) == 6

    def static_augment(self, sentences, shuffle=True):
        data_len = len(sentences)
        logger.info(get_time() + f"Raw data used in augmentation: {data_len}")
        # proportions for six classes of augmentation type
        proportion = [0.16, 0.16, 0.16, 0.16, 0.16, 0.16]
        accumulate = [int(data_len * sum(proportion[:i+1])) for i in range(6)]

        res = []
        logger.info(get_time() + "Type1...")
        for i in tqdm(range(0, accumulate[0])):
            res.extend(self.augmentors[0].augment(sentence=sentences[i]))
        logger.info(get_time() + "Type2...")
        for i in tqdm(range(accumulate[0], accumulate[1])):
            res.extend(self.augmentors[1].augment(sentence=sentences[i]))
        logger.info(get_time() + "Type3...")
        for i in tqdm(range(accumulate[1], accumulate[2])):
            res.extend(self.augmentors[2].augment(sentence=sentences[i]))
        logger.info(get_time() + "Type4...")
        for i in tqdm(range(accumulate[2], accumulate[3])):
            res.extend(self.augmentors[3].augment(sentence=sentences[i]))
        logger.info(get_time() + "Type5...")
        for i in tqdm(range(accumulate[3], accumulate[4])):
            res.extend(self.augmentors[4].augment(sentence=sentences[i]))
        logger.info(get_time() + "Type6...")
        for i in tqdm(range(accumulate[4], accumulate[5])):
            res.extend(self.augmentors[5].augment(sentence=sentences[i]))

        if shuffle:
            logger.info(get_time() + "Shuffling...")
            random.shuffle(res)

        return res


###################### main ###################

def seed(i=19260817):
    random.seed(i)
    np.random.seed(i)

if __name__ == "__main__":
    seed()
    methods = [
        ("redundant_component", RedundantComponent),
        ("missing_component", MissingComponent),
        ("structural_confusion", StructuralConfusion),
        ("improper_logicality", ImproperLogicality),
        ("improper_collocation", ImproperCollocation),
        ("improper_word_order", ImproperWordOrder),
    ]
    samples = [
        "大连是中国最美丽的城市之一",
        "各种条件的日益成熟强劲地推动了我国叉车行业的快速发展",
        "虽然对这一事件的调查正在进行，但希尔反对做出任何不利于拉扎罗的结论",
        "货物出卖人在交付产品给买受人时，经常提供服务。",
        "本中心之藏书包含台湾文学、语言、历史、文化、政治、族群关系等各领域。",
        "方法利用分组传递拥塞信息，有效地避免了分组的丢失重传。",
        "第三季总的营业费用较上年同期下降18%,为7.75亿美元.",
        "我几乎纯白色，但在夏天，我的皮毛可能会变黄",
        "正是因为民族和艺术风格的多样化，才使今天的国际艺术节如此引人注目",
        "本文主要由三个部分组成：导生制、见习生制、导生制和见习生制的历史作用",
        "春姑娘像一个温柔的妈妈。",
        "在哥伦比亚波哥大美国大使馆门外爆发了激烈的冲突。"
    ]
    for sample in samples:
        print("*" * 30)
        print("Input: ", sample)
        sample = re.sub("\s+", "", sample)
        if sample[-1] not in ['；','！','？',',','!','?','...', "。", "."]:
            sample += "。"
        for _module, _class in methods:
            method = _class()
            output = method.transform(sample)
            print("Output:", json.dumps(output, indent=2, ensure_ascii=False))
