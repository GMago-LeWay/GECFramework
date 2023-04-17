from collections import Counter
import pandas as pd
from xpinyin import Pinyin

from config import Config

from augmentors import ErrorTypeFZ as ErrorType
from augmentors.single_error import SingleCharacterSubstitution, SingleCharAugmentor
from augmentors.CLG import CLGAugmentor

def get_augmentation(name: str):
    AUG_MAP = {
        'singlecharacter': SingleCharAugmentor,
        'clg': CLGAugmentor,
    }

    assert name in AUG_MAP.keys(), 'Not support ' + name

    return AUG_MAP[name]

if __name__ == "__main__":
    from data import get_data
    augment = SingleCharacterSubstitution()
    config = Config(None, 'fangzhengtest', False).get_config()
    data_cls = get_data('fangzhengtest')(None, config)
    data = data_cls.data()
    result = augment.statistics(data)
    print(Counter(result["type"]))
    result = pd.DataFrame(result)

    p = Pinyin()

    ## type == 4 grained statistics
    def get_consonant_and_vowel(pinyin:str):
        consonants = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w", ""]
        vowels = ["a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu", "ie", "üe", "er", "an", "en", "in", "un", "ün", "ang", "eng", "ing", "ong"]
        for consonant in consonants:
            len_cons = len(consonant)
            if pinyin[:len_cons] == consonant:
                return pinyin[:len_cons], pinyin[len_cons:]
        raise NotImplementedError()

    type4 = {"consistent_consonant": [], "consistent_vowel": [], "consistent": []}
    for i, row in result.iterrows():
        if row["type"] == 4:
            for i in range(len(row["correct_word"])):
                if row["correct_word"][i] != row["wrong_word"][i]:
                    first_wrong_idx = i
                    break
            pinyin_correct = p.get_pinyin(row["correct_word"]).split('-')[first_wrong_idx]
            pinyin_wrong = p.get_pinyin(row["wrong_word"]).split('-')[first_wrong_idx]

            ## 声母与韵母的相似情况
            correct_cos, correct_vow = get_consonant_and_vowel(pinyin_correct)
            wrong_cos, wrong_vow = get_consonant_and_vowel(pinyin_wrong)

            type4["consistent_consonant"].append(1 if correct_cos == wrong_cos else 0)
            type4["consistent_vowel"].append(1 if correct_vow == wrong_vow else 0)
            type4["consistent"].append(1 if correct_vow == wrong_vow and correct_cos == wrong_cos else 0)

    print("consistent consonants %d, consistent vowels %d, consistent %d" % (sum(type4["consistent_consonant"]), sum(type4["consistent_vowel"]), sum(type4["consistent"])))


    ## type == 5 grained statistics
    type5 = {"pos": [], "dep": []}
    for i, row in result.iterrows():
        if row["type"] == 5:
            type5["pos"].append(row["pos"])
            type5["dep"].append(row["dep"])
    print(Counter(type5["pos"]))
    print(Counter(type5["dep"]))

    type6 = {"pos": [], "dep": []}
    ## type == 6 grained statistics
    for i, row in result.iterrows():
        if row["type"] == 6:
            type6["pos"].append(row["pos"])
            type6["dep"].append(row["dep"])
    print(Counter(type6["pos"]))
    print(Counter(type6["dep"]))

    result.to_excel("dataset/fangzheng_statistics.xlsx", index=None)
