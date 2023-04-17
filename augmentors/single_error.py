from collections import Counter
import os
import json
import random
import traceback
import logging
from tqdm import tqdm

import synonyms
import spacy
from pypinyin import pinyin, Style
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader, random_split

from utils import *
from config import DATA_ROOT_DIR, MODEL_ROOT_DIR

logger = logging.getLogger(__name__)

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


class SingleCharacterSubstitution:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

        # file path of augmentation tools
        self.confusion_set_path = os.path.join(DATA_ROOT_DIR, "/HybridSet/corpus/confusion.txt")
        # self.dictionary_file = "/home/liwei/workspace/datasets/Dictionary/dict.txt"
        self.dictionary_dir = "./knowledge"
        self.dictionary_file = "./knowledge/现汉+大辞海词汇.txt"

        # substitution sources
        self.dictionary_reversed_index: dict[str, list[int]] = {}
        self.dictionary_list: list[str] = []
        self._reversed_index_init()
        self.confusion_set: dict[str, list[str]] = {}
        self._confusion_set_init()

        # bert model
        self.bert_augment = None

        # return error num
        self.max_error_num = 1

        # parser
        self.chinese_parser = spacy.load("zh_core_web_trf")


    def _reversed_index_init(self):
        with open(self.dictionary_file, "r") as f:
            for idx, line in tqdm(enumerate(f.readlines())):
                word = line.split()[0].strip()
                self.dictionary_list.append(word)
                for i in range(len(word)):
                    if word[i] not in self.dictionary_reversed_index:
                        self.dictionary_reversed_index[word[i]] = [idx]
                    else:
                        self.dictionary_reversed_index[word[i]].append(idx)
        reversed_idx_file = os.path.join(self.dictionary_dir, "reversed_index.txt")
        with open(reversed_idx_file, "w") as f:
            json.dump(self.dictionary_reversed_index, f, ensure_ascii=False, indent=4)


    def _confusion_set_init(self):
        with open(self.confusion_set_path, "r") as f:
            for line in tqdm(f.readlines()):
                key, characters = line.strip().split(':')
                self.confusion_set[key] = list(characters)

    def _filter_non_character_item(self, strings: list[str]):
        final_results = []
        for s in strings:
            all_chinese = True
            for i in range(len(s)):
                if not is_chinese(uchar=s[i]):
                    all_chinese = False
                    break
            if all_chinese:
                final_results.append(s)
        return final_results

    def get_confusion_set(self, character):
        if character not in self.confusion_set:
            return []
        return self._filter_non_character_item(self.confusion_set[character])


    def get_synonym(self, word, pos_substitution=None):
        synonym_words = synonyms.nearby(word, 30)[0]
        final_result = []
        # filter
        if pos_substitution:
            for w in synonym_words:
                if len(w) != len(word):
                    continue
                # compare by character
                suitable = True
                for i in range(len(w)):
                    if i == pos_substitution and w[i] != word[i]:
                        continue
                    elif i != pos_substitution and w[i] == word[i]:
                        continue
                    else:
                        suitable = False
                        break
                if suitable:
                    final_result.append(w)
        else:
            for w in synonym_words:
                if len(w) != len(word):
                    continue
                # ensure there's only one different char
                error_count = 0
                for i in range(len(w)):
                    if w[i] != word[i]:
                        error_count += 1
                if error_count != 1:
                    continue

                final_result.append(w)
        
        return self._filter_non_character_item(final_result)


    def get_synonym_char(self, character):
        synonym_words = synonyms.nearby(character, 30)[0]
        final_result = []
        for w in synonym_words:
            if len(w) == 1:
                final_result.append(final_result)
        return self._filter_non_character_item(final_result)


    def get_word(self, word, pos_substitution):
        # get possible word by reversed index of character
        char_must_contained = []
        for i in range(len(word)):
            if i != pos_substitution:
                char_must_contained.append(word[i])

        words_indexes = None
        for char in char_must_contained:
            if char not in self.dictionary_reversed_index:
                break
            if words_indexes == None:
                words_indexes = set(self.dictionary_reversed_index[char])
            else:
                words_indexes = words_indexes & set(self.dictionary_reversed_index[char])

        words_indexes = list(words_indexes) if words_indexes else []
        final_result = []
        for index in words_indexes:
            w = self.dictionary_list[index]
            # compare
            if len(w) != len(word):
                continue
            # compare by character
            suitable = True
            for i in range(len(w)):
                if i == pos_substitution and w[i] != word[i]:
                    continue
                elif i != pos_substitution and w[i] == word[i]:
                    continue
                else:
                    suitable = False
                    break
            if suitable:
                final_result.append(w)

        return self._filter_non_character_item(final_result)
    

    def confusion_set_based_augment(self, sentence: str):
        # select position of character
        first_chinese_char_idx = 0
        while first_chinese_char_idx < len(sentence) and (not is_chinese(sentence[first_chinese_char_idx])):
            first_chinese_char_idx += 1
        if first_chinese_char_idx == len(sentence):
            return []

        sample_num = min(len(sentence) - first_chinese_char_idx, self.max_error_num)
        substitute_idx = random.sample(list(range(first_chinese_char_idx, len(sentence))), sample_num)

        # generate
        res = []
        for pos in substitute_idx:
            if not is_chinese(sentence[pos]):
                continue
            # get confusion set
            confusion_char_list = self.get_confusion_set(sentence[pos])
            if confusion_char_list == []:
                continue
            else:
                selected_char = confusion_char_list[random.randint(0, len(confusion_char_list)-1)]
                ## final check
                if not is_chinese(selected_char):
                    continue
                src = sentence[:pos] + selected_char + sentence[pos+1:]
                res.append({"text": src, "label": sentence, "rule": "confusion_set"})

        return res
    
    def synonym_char_based_augment(self, sentence: str):
        # select position of character
        first_chinese_char_idx = 0
        while first_chinese_char_idx < len(sentence) and (not is_chinese(sentence[first_chinese_char_idx])):
            first_chinese_char_idx += 1
        if first_chinese_char_idx == len(sentence):
            return []

        sample_num = min(len(sentence) - first_chinese_char_idx, self.max_error_num)
        substitute_idx = random.sample(list(range(first_chinese_char_idx, len(sentence))), sample_num)

        # generate
        res = []
        for pos in substitute_idx:
            if not is_chinese(sentence[pos]):
                continue
            # get synonym set
            synonym_char_list = self.get_synonym_char(sentence[pos])
            if synonym_char_list == []:
                continue
            else:
                selected_char = synonym_char_list[random.randint(0, len(synonym_char_list)-1)]
                ## final check
                if not is_chinese(selected_char):
                    continue
                src = sentence[:pos] + selected_char + sentence[pos+1:]
                res.append({"text": src, "label": sentence, "rule": "synonym_char"})
        return res
    
    def synonym_based_augment(self, sentence: str):
        doc = self.chinese_parser(sentence)
        # filter the word with >=2 syllables
        cumulative_idx = 0
        record_of_word = []      # [word, word_position]
        for token in doc:
            token_len = len(token.text)
            if token_len > 1:
                record_of_word.append([token.text, cumulative_idx])
            cumulative_idx += token_len
        if record_of_word == []:
            return []

        # random word
        sample_num = min(len(record_of_word), self.max_error_num)
        word_idx = random.sample(list(range(0, len(record_of_word))), sample_num)

        # generate
        res = []
        for idx in word_idx:
            word_to_augment, word_position = record_of_word[idx]
            words_augmented = self.get_synonym(word_to_augment)
            if words_augmented == []:
                continue
            else:
                selected_word = words_augmented[random.randint(0, len(words_augmented)-1)]
                check_success = True
                for character in list(selected_word):
                    if not is_chinese(character):
                        check_success = False
                        break
                if not check_success:
                    continue
                src = sentence[:word_position] + selected_word + sentence[word_position+len(selected_word):]
                res.append({"text": src, "label": sentence, "rule": "synonym_word"})
        return res
    
    def word_based_augment(self, sentence: str):
        doc = self.chinese_parser(sentence)
        # filter the word with >=2 syllables
        cumulative_idx = 0
        record_of_word = []      # [word, word_position]
        for token in doc:
            token_len = len(token.text)
            if token_len > 1:
                record_of_word.append([token.text, cumulative_idx])
            cumulative_idx += token_len
        if record_of_word == []:
            return []

        # random word
        src = str(sentence)
        sample_num = min(len(record_of_word), self.max_error_num)
        word_idx = random.sample(list(range(0, len(record_of_word))), sample_num)

        # generate
        res = []
        for idx in word_idx:
            word_to_augment, word_position = record_of_word[idx]
            words_augmented = []
            for j in range(len(word_to_augment)):
                words_augmented.extend(self.get_word(word=word_to_augment, pos_substitution=j))
            if words_augmented == []:
                continue
            else:
                selected_word = words_augmented[random.randint(0, len(words_augmented)-1)]
                check_success = True
                for character in list(selected_word):
                    if not is_chinese(character):
                        check_success = False
                        break
                if not check_success:
                    continue
                src = sentence[:word_position] + selected_word + sentence[word_position+len(selected_word):]
                res.append({"text": src, "label": sentence, "rule": "substitute_word"})
        return res

    def bert_single_char_word_augment(self, sentences):
        assert type(sentences) == list
        if self.bert_augment == None:
            self.bert_augment = BertForAugmentation(self.args, self.config)

        bert_aug_list = []
        aug_position = []
        _max_index_in_sentence = 500
        temp_labels = []
        # type5: single character error
        for sentence in sentences:
            if len(sentence) > self.config.text_cut:
                continue
            doc = self.chinese_parser(sentence)
            # filter the word with 1 syllables
            cumulative_idx = 0
            record_of_word = []      # [word, word_position]
            for token in doc:
                token_len = len(token.text)
                if cumulative_idx + token_len > _max_index_in_sentence:  # avoid position being truncated in bert
                    break
                if token_len == 1 and (token.pos_ not in ["PUNCT", "X"]):
                    record_of_word.append([token.text, cumulative_idx])
                cumulative_idx += token_len
            if record_of_word == []:
                continue          

            # random word, to bert augment input
            word_to_augment, word_position = record_of_word[random.randint(0, len(record_of_word)-1)]
            bert_aug_list.append(sentence)
            aug_position.append(word_position)
            temp_labels.append(sentence)

        bert_augment_results, bert_augment_flags = self.bert_augment.augment(bert_aug_list, aug_position)
        assert len(bert_augment_results) == len(temp_labels)
        res = []
        for i in range(len(bert_augment_flags)):
            if bert_augment_flags[i] == True:
                res.append({"text": bert_augment_results[i], "label": temp_labels[i], "rule": "bert_single_char_word"})
        return res

    def bert_single_char_augment(self, sentences):
        assert type(sentences) == list
        if self.bert_augment == None:
            self.bert_augment = BertForAugmentation(self.args, self.config)

        bert_aug_list = []
        aug_position = []
        _max_index_in_sentence = 500
        temp_labels = []

        for sentence in sentences:
            if len(sentence) > self.config.text_cut:
                continue
            # select position of character
            first_chinese_char_idx = 0
            while first_chinese_char_idx < len(sentence) and (not is_chinese(sentence[first_chinese_char_idx])):
                first_chinese_char_idx += 1
            if first_chinese_char_idx == len(sentence) or first_chinese_char_idx > _max_index_in_sentence: # avoid position being truncated in bert
                continue
            pos = first_chinese_char_idx
            for j in range(10):
                random_pos = random.randint(first_chinese_char_idx, len(sentence) - 1)
                if is_chinese(sentence[random_pos]) and random_pos <= _max_index_in_sentence:
                    pos = random_pos
                    break
            bert_aug_list.append(sentence)
            aug_position.append(pos)
            temp_labels.append(sentence)

        bert_augment_results, bert_augment_flags = self.bert_augment.augment(bert_aug_list, aug_position)
        assert len(bert_augment_results) == len(temp_labels)
        res = []
        for i in range(len(bert_augment_flags)):
            if bert_augment_flags[i] == True:
                res.append({"text": bert_augment_results[i], "label": temp_labels[i], "rule": "bert_single_char"})
        return res


    def statistics(self, data):
        """
        error type:
        0: not single character substitution error
        1: character in confusion set
        2: character in synonym characters
        3: word in synonym words
        4: construct a new word
        5: single character word error not included above.
        6: others.
        """
        result = {"text": [], "label": [], "wrong_word": [], "correct_word": [], "type": [], "pos": [], "dep": []}

        for item in tqdm(data):
            for error in item["errors"]:
                wrong_word = error["wrong_word"]
                correct_word = error["correct_word"]
                # copy information
                result["text"].append(item["text"])
                result["label"].append(item["label"])
                result["wrong_word"].append(wrong_word)
                result["correct_word"].append(correct_word)
                result["pos"].append(error["pos"])
                result["dep"].append(error["dep"])

                # confirm that it's a single character
                if len(wrong_word) != len(correct_word):
                    result["type"].append(0)
                    continue
                count = 0
                first_wrong_idx = None
                for i in range(len(correct_word)):
                    if correct_word[i] != wrong_word[i]:
                        if first_wrong_idx is None:
                            first_wrong_idx = i
                        count += 1
                if count != 1:
                    result["type"].append(0)
                    continue

                # select the wrong character and the corresponded correct character
                wrong_char, correct_char = wrong_word[first_wrong_idx], correct_word[first_wrong_idx]

                if wrong_char in self.get_confusion_set(character=correct_char):
                    result["type"].append(1)
                elif wrong_char in self.get_synonym_char(character=correct_char):
                    result["type"].append(2)
                elif wrong_word in self.get_synonym(word=correct_word, pos_substitution=first_wrong_idx):
                    result["type"].append(3)
                elif wrong_word in self.get_word(word=correct_word, pos_substitution=first_wrong_idx):
                    result["type"].append(4)
                elif len(wrong_word) == 1:
                    result["type"].append(5)
                else:
                    result["type"].append(6)

        return result


class SingleCharAugmentor:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config
        self.single_augmentor = SingleCharacterSubstitution(args, config)

    def static_augment(self, sentences, shuffle=True):
        data_len = len(sentences)
        logger.info(get_time() + f"Raw data used in augmentation: {data_len}")
        # proportions for six classes of augmentation type
        proportion = [0.06, 0.0, 0.12, 0.40, 0.17, 0.25]
        accumulate = [int(data_len * sum(proportion[:i+1])) for i in range(6)]


        res = []
        logger.info(get_time() + "Type1...")
        for i in tqdm(range(0, accumulate[0])):
            res.extend(self.single_augmentor.confusion_set_based_augment(sentence=sentences[i]))
        logger.info(get_time() + "Type2...")
        for i in tqdm(range(accumulate[0], accumulate[1])):
            res.extend(self.single_augmentor.synonym_char_based_augment(sentence=sentences[i]))
        logger.info(get_time() + "Type3...")
        for i in tqdm(range(accumulate[1], accumulate[2])):
            res.extend(self.single_augmentor.synonym_based_augment(sentence=sentences[i]))
        logger.info(get_time() + "Type4...")
        for i in tqdm(range(accumulate[2], accumulate[3])):
            res.extend(self.single_augmentor.word_based_augment(sentence=sentences[i]))

        logger.info(get_time() + "Type5...")
        res.extend(self.single_augmentor.bert_single_char_word_augment(sentences=sentences[accumulate[3]:accumulate[4]]))
        logger.info(get_time() + "Type6...")
        res.extend(self.single_augmentor.bert_single_char_augment(sentences=sentences[accumulate[4]:accumulate[5]]))

        if shuffle:
            logger.info(get_time() + "Shuffling...")
            random.shuffle(res)

        return res
    
    def dynamic_augment(self, sentence):
        raise NotImplementedError()
