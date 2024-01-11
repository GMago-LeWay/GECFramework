import torch
import torch.utils.data
import random
import copy
import numpy as np
import math
from scipy.stats import poisson
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoConfig
import logging

logger = logging.getLogger(__name__)

IGNORE_ID=-100

LAST_DETECTION_ERROR_NUM=0
SEQUENCE_MATCH_ERROR_NUM=0
ERROR_NUM=0

def rindex(lst, val, start=None):
    if start is None:
        start = len(lst) - 1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i
    return -1


def index_in_list(lst, val, start=None):
    if start is None:
        start = 0
    for i in range(start, len(lst)):
        if lst[i] == val:
            return i
    return -1


class TokenizerBasedTextEditProcessor:
    def __init__(self, tokenizer, without_insert=False, max_sequence_length=None, task_mode=None) -> None:
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.task_mode = task_mode
        # if self.task_mode in ['infer_train', 'eval_train']:
        #     logger.info("You are using infer train mode or eval train mode, split sentence to max_seq_len by default.")
        # else:
        logger.info("while the max_length is not set, sentence split sentence to max_seq_len // 3 by default.")
        self.keep_label = '$KEEP'
        self.insert_label = '$INSERT' # Notice that 'insert' means insertion AFTER the current token
        self.error_label = '$ERROR'
        self.ignore_label = '$IGNORE'
        self.edit_label_map = {
            self.ignore_label: IGNORE_ID,
            self.keep_label: 0,
            self.error_label: 1,
            self.insert_label: 2,
        }
        self.edit_label_id_map = {
            0: self.keep_label,
            1: self.error_label,
            2: self.insert_label,
        }
        self.without_insert = without_insert
        # self.delete_label = '$DELETE'
        # self.replace_label = '$REPLACE'
        self.marker1 = ['。', '？', '！', '；', '…', '?', '!', '.', '\n']
        self.marker2 = ['”', '’', '"', '\\', '、', ',', '，']
        self.marker1_ids = self.tokenizer.convert_tokens_to_ids(self.marker1)
        self.marker2_ids = self.tokenizer.convert_tokens_to_ids(self.marker2)

    def split_sentence(self, sentence: str, max_sentence_length: int, enable_warning = False) -> List[int]:
        '''
        Use tokenizer to split sentence into a list of tokens
        '''
        reserved_length = 12
        if max_sentence_length is None:
            if self.max_sequence_length:
                # if self.task_mode in ['infer_train', 'eval_train']:
                #     max_sentence_length = self.max_sequence_length - reserved_length
                # else:
                max_sentence_length = self.max_sequence_length // 3
        tokens = self.tokenizer.encode(sentence)
        if max_sentence_length < len(tokens):
            if enable_warning:
                logger.info(f"Warning: The sentence token length {len(tokens)}, will be cut to {max_sentence_length}")
            return tokens[:max_sentence_length-1] + tokens[-1:]
        else:
            return tokens
    
    def limited_split_sentence(self, src: str, tgt: str, length: int) -> Tuple[List[int], List[int], Tuple[str, int, int, int, int]]:
        '''
        Use tokenizer to split sentence into a list of tokens, limit the sentence in a fixed length.
        '''
        src_tokens = self.split_sentence(src, None)
        tgt_tokens = self.split_sentence(tgt, None)
        if self.without_insert:
            diffs = self.align_without_insert(src_tokens=src_tokens, tgt_tokens=tgt_tokens)
        else:
            r = SequenceMatcher(None, src_tokens, tgt_tokens)
            diffs = r.get_opcodes()

        # did not reach limit length
        if len(src_tokens) <= length and len(tgt_tokens) <= length:
            edit_diffs = [diff for diff in diffs if diff[0] in ['replace', 'insert', 'delete']]
            return src_tokens, tgt_tokens, edit_diffs
        
        # reach the limit length
        # get last aligned spans that have start points less than the limit length.
        last_equal_align = None
        truncate_diff_idx = None
        for diff_idx, diff in enumerate(diffs):
            if diff[0] == 'equal':
                if length > diff[1] and length > diff[3]:
                    last_equal_align = diff
                    truncate_diff_idx = diff_idx
            if length < diff[1] and length < diff[3]:
                break
        assert last_equal_align is not None, f"Did not find a aligned span before length limit {length}"

        # find marker for truncate, first find in marker group 1, next find in marker group 2:
        # 1) first find max length limit, identify the span of finding markers.
        limit_length_of_src_tokens = min(last_equal_align[2], length, length-last_equal_align[3]+last_equal_align[1])
        # 2) identify the truncating index in the span
        truncate_idx = None
        for idx in range(limit_length_of_src_tokens-1, last_equal_align[1]-1, -1):
            if src_tokens[idx] in self.marker1_ids:
                truncate_idx = idx
                break
        if truncate_idx is None:
            for idx in range(limit_length_of_src_tokens-1, last_equal_align[1]-1, -1):
                if src_tokens[idx] in self.marker2_ids:
                    truncate_idx = idx
                    break
        # 3) if there is no marker, truncate at last aligned token
        if truncate_idx is None:
            truncate_idx = limit_length_of_src_tokens
        # 4) truncate
        truncate_length_in_last_span = truncate_idx - last_equal_align[1]
        src_tokens = src_tokens[:last_equal_align[1]+truncate_length_in_last_span]
        tgt_tokens = tgt_tokens[:last_equal_align[3]+truncate_length_in_last_span]
        new_diffs = diffs[:truncate_diff_idx]
        edit_diffs = [diff for diff in new_diffs if diff[0] in ['replace', 'insert', 'delete']]
        return src_tokens + [self.tokenizer.eos_token_id], tgt_tokens + [self.tokenizer.eos_token_id], edit_diffs

    def align_without_insert(self, src_tokens: List[int], tgt_tokens: List[int]) -> List[Tuple[str, int, int, int, int]]:
        '''
        Extract edit list for source text transforming to target text. for insert (src_idx, src_idx, tgt_idx1, tgt_idx2), turn to ()
        (src_idx_start, src_idx_end, tgt_idx_start, tgt_idx_end)
        '''
        r = SequenceMatcher(None, src_tokens, tgt_tokens)
        diffs = r.get_opcodes()
        diff_processed_flag = [False]*len(diffs)    # avoid repeat edit
        edits = []
        assert diffs[0][0] == 'equal', "Required item at start is 'equal'."
        for i in range(len(diffs)-1, -1, -1):
            diff = diffs[i]
            if diff_processed_flag[i]:
                continue
            if diff[0] in ['replace', 'delete', 'equal']:
                edits.append(diff)
                continue
            assert diff[0] == 'insert', "Other unlegalled label exists."
            assert diffs[i-1][0] == 'equal', "Matcher should have merge adjacent edits. Please check"
            if diffs[i-1][2] - diffs[i-1][1] == 1:      # last equal span length==1
                if i-2>=0:   # merge two aligned spans
                    assert diffs[i-2][0] != 'equal' 
                    edits.append(('replace', diffs[i-2][1], diffs[i][2], diffs[i-2][3], diffs[i][4]))
                    diff_processed_flag[i-1] = True
                    diff_processed_flag[i-2] = True
                else:    # there is no more prefix edit
                    edits.append(('replace', diffs[i-1][1], diffs[i][2], diffs[i-1][3], diffs[i][4]))
                    diff_processed_flag[i-1] = True
            else:                                       # last span length > 1
                # expand edit alignment by 1
                edits.append(('replace', diffs[i][1]-1, diffs[i][2], diffs[i][3]-1, diffs[i][4]))
                edits.append(('equal', diffs[i-1][1], diffs[i-1][2]-1, diffs[i-1][3], diffs[i-1][4]-1))
                diff_processed_flag[i-1] = True

        edits.reverse()
        return edits

    def edit(self, src_tokens: List[int], tgt_tokens: List[int]) -> List[Tuple[str, int, int, int, int]]:
        '''
        Extract edit list for source text transforming to target text.
        (src_idx_start, src_idx_end, tgt_idx_start, tgt_idx_end)
        '''
        if self.without_insert:
            diffs = self.align_without_insert(src_tokens=src_tokens, tgt_tokens=tgt_tokens)
        else:
            r = SequenceMatcher(None, src_tokens, tgt_tokens)
            diffs = r.get_opcodes()
        
        return [diff for diff in diffs if diff[0] in ['replace', 'insert', 'delete']]
    
    def edit_labels(self, edits: List[Tuple[str, int, int, int, int]], src_tokens: List[int], tgt_tokens: List[int]) -> List[str]:
        '''
        Sequence labeling of edit label using edit list.
        Current Label: $KEEP, $APPEND, $ERROR
        '''
        labels = [self.keep_label] * len(src_tokens)
        for edit in edits:
            _, i1, i2, j1, j2 = edit
            if i1==i2:     # insert [j1, j2) at i1
                assert i1>0, f"{i1} {i2}, {j1} {j2}"
                labels[i1-1] = self.insert_label
            else:  # i1<i2
                for i in range(i1, i2):
                    labels[i] = self.error_label
        return labels


class GLMDataProcessor:
    def __init__(self, tokenizer, args, config) -> None:
        self.tokenizer = tokenizer
        self.model_config = AutoConfig.from_pretrained(config.pretrained_model, trust_remote_code=True)
        self.max_length = self.model_config.max_sequence_length
        # .sop_token/.eop_token/.sop_token_id/.eop_token_id
        assert config.num_labels in [2, 3]
        if config.num_labels == 2:
            self.without_insert = True
        else:
            self.without_insert = False
        self.edit_extractor = TokenizerBasedTextEditProcessor(self.tokenizer, without_insert=self.without_insert, max_sequence_length=self.max_length, task_mode=args.task_mode)
        self.args = args
        self.config = config
        # edit settings
        self.keep_label = self.edit_extractor.keep_label
        self.insert_label = self.edit_extractor.insert_label
        self.error_label = self.edit_extractor.error_label
        self.ignore_label = self.edit_extractor.ignore_label
        self.edit_label_map = self.edit_extractor.edit_label_map
        self.edit_label_id_map = self.edit_extractor.edit_label_id_map
        # detection part template for input
        self.detection_prefix = config.prompt
        self.detection_prefix_tokens = self.tokenizer.encode(self.detection_prefix)
        self._loss_ignore_id = IGNORE_ID
    
    def from_edit_to_glm_example(self, edits: List[Tuple[str, int, int, int, int]], original_src_tokens: List[int], original_tgt_tokens: List[int]):
        position_ids = np.ones(len(original_tgt_tokens)+1, dtype=int)        # +1 exclude bug when tgt_start = tgt_end = len(tgt_tokens)
        for _, _, _, tgt_start, tgt_end in edits:
            if tgt_start==tgt_end: # DELETE operation. In the span a [MASK] will be added, this position id should be reserved.
                position_ids[tgt_start] += 1
            else:
                position_ids[tgt_start + 1: tgt_end] = 0
        position_ids = np.cumsum(position_ids) - 1

        # target_tokens: input end ; targets: output end
        target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
        for _, src_start, src_end, tgt_start, tgt_end in edits:
            target_tokens.append([self.tokenizer.sop_token_id])
            if tgt_start == tgt_end: # DELETE operation position, reserved position id
                target_position_ids.append([position_ids[tgt_start]-1])
            else:
                span_tokens = copy.deepcopy(original_tgt_tokens[tgt_start: tgt_end])
                target_tokens.append(span_tokens)
                targets.append(original_tgt_tokens[tgt_start: tgt_end])
                target_position_id = position_ids[tgt_start: tgt_end]
                target_position_ids.append(target_position_id)
                target_position_ids.append([target_position_id[0]])
            targets.append([self.tokenizer.eop_token_id])

            target_block_position_ids.append(np.arange(1, tgt_end - tgt_start + 2, dtype=int))

        source_tokens, source_position_ids = [], []
        last, current_length = 0, 0
        mask_id = self.tokenizer.mask_token_id
        for _, src_start, src_end, tgt_start, tgt_end in edits:
            # local_spans.append((current_length, current_length + start - last))
            if last != tgt_start:
                source_tokens.append(original_tgt_tokens[last: tgt_start])
                source_position_ids.append(position_ids[last: tgt_start])
            source_tokens.append([mask_id])
            if tgt_start == tgt_end: # DELETE operation position, reserved position id
                source_position_ids.append([position_ids[tgt_start]-1])
            else:
                source_position_ids.append([position_ids[tgt_start]])
            current_length += tgt_start - last + 1
            last = tgt_end
        if last < len(original_tgt_tokens):
            # local_spans.append((current_length, current_length + len(original_src_tokens) - last))
            source_tokens.append(original_tgt_tokens[last:])
            source_position_ids.append(position_ids[last:len(original_tgt_tokens)])
        source_length = sum(map(len, source_tokens))

        tokens = np.concatenate(source_tokens + target_tokens)
        targets = np.concatenate([source_length*[self._loss_ignore_id]] + targets)
        # loss_masks = np.ones(len(tokens), dtype=np.long)
        # loss_masks[:source_length] = 0
        position_ids = np.concatenate(source_position_ids + target_position_ids)
        block_position_ids = np.concatenate(
            [np.zeros(source_length, dtype=int)] + target_block_position_ids)
        position_ids = np.stack([position_ids, block_position_ids], axis=0)
        return {
            'input_ids': tokens, 
            'target_ids': targets, 
            'position_ids': position_ids,
            'source_length': source_length,
        }
    
    def from_edit_label_to_masked_example(self, edit_labels: List[str], src_tokens: List[int]):
        if len(edit_labels) != len(src_tokens):
            assert len(edit_labels) > len(src_tokens)
            edit_labels = edit_labels[:len(src_tokens)]
        global LAST_DETECTION_ERROR_NUM
        mask_spans = []
        # hard limit: last token is always correct (<eos>)
        if edit_labels[-1] != self.keep_label:
            edit_labels[-1] = self.keep_label
            LAST_DETECTION_ERROR_NUM += 1
            logger.info(f"({LAST_DETECTION_ERROR_NUM})Warning: Last position is not $KEEP when using edit labels for converting to masked sentence. It is illegal, set it to $KEEP.")

        last_correct_token_idx = -1
        # get mask spans
        for i, edit_label in enumerate(edit_labels):
            if edit_label == self.error_label:
                # ERROR label will 
                pass
            else:
                # KEEP or INSERT means token i is correct, if the precedence token is not correct, then mask the span.
                # This means when encounting KEEP or INSERT, the edit labels whose index < i can be processed perfectly through cases.
                # case 1: the last token is not i-1 (ERROR spans exists, including like ...K,[E,E,E],K... and ...I,[E,E],K...)
                if last_correct_token_idx != i-1:
                    mask_spans.append((last_correct_token_idx+1, i))
                # case 2: the last token is i-1 but last label is INSERT (like ...I,K... or ...I,I... )
                else:
                    if last_correct_token_idx > 0 and edit_labels[i-1]==self.insert_label:
                        mask_spans.append((i, i))
                last_correct_token_idx = i
        
        position_ids = np.ones(len(src_tokens)+1, dtype=int)
        for start, end in mask_spans:
            if start==end: # INSERT operation. In the span a [MASK] will be added, this position id should be reserved.
                position_ids[start] += 1
            else:
                position_ids[start + 1: end] = 0
        position_ids = np.cumsum(position_ids) - 1

        source_tokens, source_position_ids = [], []
        mask_tokens_position = []
        last, current_length = 0, 0
        mask_id = self.tokenizer.mask_token_id
        for start, end in mask_spans:
            # local_spans.append((current_length, current_length + start - last))
            source_tokens.append(src_tokens[last: start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last: start])
            mask_tokens_position.append(sum(map(len, source_tokens))-1)
            if start == end: # INSERT operation position, reserved position id
                source_position_ids.append([position_ids[start]-1])
            else:
                source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(src_tokens):
            # local_spans.append((current_length, current_length + len(original_src_tokens) - last))
            source_tokens.append(src_tokens[last:])
            source_position_ids.append(position_ids[last:len(src_tokens)])

        tokens = np.concatenate(source_tokens, dtype=int)
        source_length = len(tokens)
        position_ids = np.stack([np.concatenate(source_position_ids), np.zeros(source_length, dtype=int)], axis=0)
            
        return {
            'input_ids': tokens,
            'position_ids': position_ids,
            'source_length': source_length,
            'mask_positions': mask_tokens_position
        }
            
    def from_edit_label_to_glm_infer_example(self, edit_labels: List[str], src_tokens: List[int]):
        masked_example = self.from_edit_label_to_masked_example(edit_labels, src_tokens)
        mask_tokens_position = masked_example['mask_positions']
        source_tokens = masked_example['input_ids']
        source_position_ids = masked_example['position_ids']
        source_length = masked_example['source_length']

        if mask_tokens_position:
            source_tokens = np.concatenate((source_tokens, [self.tokenizer.sop_token_id]), dtype=int)
            # loss_masks = np.ones(len(tokens), dtype=np.long)
            # loss_masks[:source_length] = 0
            source_position_ids = np.concatenate((source_position_ids, [[mask_tokens_position[0]], [1]]), axis=1)
            
        return {
            'input_ids': source_tokens, 
            'target_ids': np.array([self._loss_ignore_id] * len(source_tokens), dtype=int), 
            'position_ids': source_position_ids,
            'source_length': source_length,
        }
    
    def from_detection_results_to_augmented_glm_example(self, detections: List[str], detection_labels: List[str], src_tokens: List[int], tgt_tokens: List[int]):
        global SEQUENCE_MATCH_ERROR_NUM
        # merge labels to detection results
        assert len(detections) == len(detection_labels), "Maybe uncompatible detection results are applied on this dataset."
        new_detections = []
        for i in range(len(detections)):
            if detection_labels[i] != self.keep_label:
                new_detections.append(detection_labels[i])
            else:
                new_detections.append(detections[i])
        masked_example = self.from_edit_label_to_masked_example(new_detections, src_tokens)
        mask_positions = masked_example['mask_positions']
        masked_tokens = masked_example['input_ids'].tolist()
        source_position_ids = masked_example['position_ids'].tolist()
        source_length = masked_example['source_length']
        # aligned by mask
        try:
            r = SequenceMatcher(None, masked_tokens, tgt_tokens)
            diffs = r.get_opcodes()

            generation_edits = []
            # check if every span corresponding to one MASK
            for diff in diffs:
                if diff[0] == 'equal':
                    continue
                else:
                    generation_edits.append(diff)
            assert len(generation_edits) == len(mask_positions), f"source_tokens: {src_tokens}\ntgt_tokens: {tgt_tokens}\nmasked_tokens: {masked_tokens}\n auto_edits: {generation_edits}"
        except:
            SEQUENCE_MATCH_ERROR_NUM += 1
            logger.info(f"Sequence Matcher Failed({SEQUENCE_MATCH_ERROR_NUM}), trying to re-aligned...")
            generation_edits = []
            # traditional match
            idx1, idx2 = 0, 0
            while idx1 < len(masked_tokens) and idx2 < len(tgt_tokens):
                if masked_tokens[idx1] == tgt_tokens[idx2]:
                    idx1 += 1
                    idx2 += 1
                else:
                    assert masked_tokens[idx1] == self.tokenizer.mask_token_id, f"source_tokens: {src_tokens}\ntgt_tokens: {tgt_tokens}\nmasked_tokens: {masked_tokens}\nsource: {self.tokenizer.decode(src_tokens)}\n target: {self.tokenizer.decode(tgt_tokens)}\n masked: {self.tokenizer.decode(masked_tokens)}"
                    # notice that the last token would not be [MASK] because the last edit label will be set to "$KEEP" 
                    diff_start = idx2
                    diff_end = idx2
                    while tgt_tokens[diff_end] != masked_tokens[idx1+1]:
                        diff_end += 1
                    # (legacy) check if next token is equal or masked
                    # It is a temp processed method, more complicate situation would not be processed.
                    # if idx1 + 2 < len(masked_tokens) and masked_tokens[idx1+2] != self.tokenizer.mask_token_id:
                    #     if diff_end + 1 < len(tgt_tokens):
                    #         if masked_tokens[idx1+2] != tgt_tokens[diff_end+1]:
                    #             logger.info(f"Hard to align: \nsource: {self.tokenizer.decode(src_tokens)}\n target: {self.tokenizer.decode(tgt_tokens)}\n masked: {self.tokenizer.decode(masked_tokens)}")
                    #             diff_end += 1
                    #             while tgt_tokens[diff_end] != masked_tokens[idx1+1]:
                    #                 diff_end += 1
                    #             logger.info(f"Hard Alignment: \nmask_id: {len(generation_edits)}\nAlignment: {self.tokenizer.decode(tgt_tokens[diff_start:diff_end])}")

                    generation_edits.append(('replace', idx1, idx1+1, diff_start, diff_end))
                    idx1 += 1
                    idx2 = diff_end
            assert idx1 == len(masked_tokens) and idx2 == len(tgt_tokens), f"source_tokens: {src_tokens}\ntgt_tokens: {tgt_tokens}\nmasked_tokens: {masked_tokens}\n auto_edits: {generation_edits}"
            assert len(generation_edits) == len(mask_positions), f"source_tokens: {src_tokens}\ntgt_tokens: {tgt_tokens}\nmasked_tokens: {masked_tokens}\n auto_edits: {generation_edits}"
            

        # target_tokens: input end ; targets: output end
        target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
        for diff_label, src_start, src_end, tgt_start, tgt_end in generation_edits:
            # check
            assert src_end-src_start==1 and masked_tokens[src_start] == self.tokenizer.mask_token_id, "MASK Position uncompatible to edits, please check if all true edit labels are included in examples."
            # add target
            target_tokens.append([self.tokenizer.sop_token_id] + tgt_tokens[tgt_start:tgt_end])
            targets.append(tgt_tokens[tgt_start:tgt_end] + [self.tokenizer.eop_token_id])
            target_position_ids.append([src_start]*(1 + tgt_end - tgt_start))
            target_block_position_ids.append(np.arange(1, tgt_end - tgt_start + 2, dtype=int))
        
        tokens = np.concatenate([masked_tokens] + target_tokens)
        targets = np.concatenate([source_length*[self._loss_ignore_id]] + targets)
        position_ids = np.concatenate([source_position_ids[0]] + target_position_ids)
        block_position_ids = np.concatenate(
            [np.zeros(source_length, dtype=int)] + target_block_position_ids)
        position_ids = np.stack([position_ids, block_position_ids], axis=0)
        return {
            'input_ids': tokens, 
            'target_ids': targets, 
            'position_ids': position_ids,
            'source_length': source_length,
        }
    
    def add_detection_prefix(self, glm_example: Dict, src_tokens: List[int], edit_labels: List[str]):
        detection_tokens = np.concatenate([self.detection_prefix_tokens, src_tokens])
        prefix_length = len(detection_tokens)
        full_src_tokens = np.concatenate([detection_tokens, glm_example['input_ids']])
        full_target_tokens = np.concatenate([[self._loss_ignore_id] * prefix_length, glm_example['target_ids']])
        full_position_ids = np.concatenate([np.arange(0, prefix_length, dtype=int), glm_example['position_ids'][0, :] + prefix_length])
        full_block_ids = np.concatenate([[0]*prefix_length, glm_example['position_ids'][1, :]])
        edit_label_ids = [self.edit_label_map[item] for item in edit_labels]
        if self.without_insert:
            assert max(edit_label_ids) <= 1
        return {
            'input_ids': full_src_tokens,
            'target_ids': full_target_tokens,
            'detection_labels': np.array([self._loss_ignore_id]*len(self.detection_prefix_tokens) + edit_label_ids + [self._loss_ignore_id]*len(glm_example['input_ids']), dtype=int),
            'position_ids': np.stack([full_position_ids, full_block_ids], axis=0),
            'source_length': glm_example['source_length'] + prefix_length,
            'prefix_length': prefix_length,
            'prefix_prompt_length': len(self.detection_prefix_tokens),
        }

    def convert_gec_sentence_pair_to_example_using_detections(self, src: str, tgt: str, edit_prediction_ids: List[int], max_sentence_length: int):
        # use detection predictions as augment source to construct parallel training data
        global ERROR_NUM
        src_tokens, tgt_tokens, edits = self.edit_extractor.limited_split_sentence(src, tgt, max_sentence_length)
        edit_labels = self.edit_extractor.edit_labels(edits=edits, src_tokens=src_tokens, tgt_tokens=tgt_tokens)
        edit_predictions = [self.edit_label_id_map[i] for i in edit_prediction_ids]
        try:
            example = self.from_detection_results_to_augmented_glm_example(detections=edit_predictions[:len(edit_labels)], detection_labels=edit_labels, src_tokens=src_tokens, tgt_tokens=tgt_tokens)
            train_example = self.add_detection_prefix(example, src_tokens=src_tokens, edit_labels=edit_labels)
        except:
            ERROR_NUM += 1
            logger.info(f"({ERROR_NUM})ERROR Occured while converting data.")
            # example = {
            #     'input_ids': np.array([], dtype=int), 
            #     'target_ids': np.array([], dtype=int), 
            #     'position_ids': np.array([[],[]], dtype=int),
            #     'source_length': 0,
            # }
            # train_example = self.add_detection_prefix(example, src_tokens=src_tokens, edit_labels=[self.ignore_label]*len(edit_labels))
            train_example =  self.convert_gec_sentence_pair_to_example(src=src, tgt=tgt, max_sentence_length=max_sentence_length)
        assert train_example['input_ids'].dtype == train_example['input_ids'].dtype == train_example['detection_labels'].dtype == train_example['position_ids'].dtype == int
        assert type(train_example['source_length']) == type(train_example['prefix_length']) == type(train_example['prefix_prompt_length']) == int
        return train_example
    
    def convert_gec_sentence_pair_to_example_using_masked_text(self, src: str, tgt: str, masked_text: Union[str, List[str]], max_sentence_length: int = 100):
        '''
        use masked text or masked words list as augment source to construct parallel training data
        '''
        # use masked text as augment source to construct parallel training data
        # convert to detection predictions in the condition of current tokenizer
        # align source text and masked text
        original_src_tokens = self.edit_extractor.split_sentence(src, self.max_length, enable_warning=True)
        if type(masked_text) == str:
            masked_src_tokens = self.edit_extractor.split_sentence(masked_text, self.max_length, enable_warning=True)
        elif type(masked_text) == list:
            masked_src_tokens = self.tokenizer.convert_tokens_to_ids(masked_text)[:self.max_length-2]
            # add special tokens
            if original_src_tokens[0] in self.tokenizer.all_special_ids:
                masked_src_tokens.insert(0, original_src_tokens[0])
            if original_src_tokens[-1] in self.tokenizer.all_special_ids:
                masked_src_tokens.append(original_src_tokens[-1])
        else:
            raise NotImplementedError()

        mask_positions = []
        for i, token in enumerate(masked_src_tokens):
            if token == self.tokenizer.mask_token_id:
                mask_positions.append(i)
        global SEQUENCE_MATCH_ERROR_NUM, ERROR_NUM
        try:
            try:
                r = SequenceMatcher(None, original_src_tokens, masked_src_tokens)
                diffs = r.get_opcodes()

                generation_edits = []
                # check if every span corresponding to one MASK
                for diff in diffs:
                    if diff[0] == 'equal':
                        continue
                    else:
                        generation_edits.append(diff)
                assert len(generation_edits) == len(mask_positions), f"source_tokens: {original_src_tokens}\nmasked_tokens: {masked_src_tokens}\n auto_edits: {generation_edits}"
            except:
                SEQUENCE_MATCH_ERROR_NUM += 1
                logger.info(f"Sequence Matcher Failed({SEQUENCE_MATCH_ERROR_NUM}), trying to re-aligned...")
                generation_edits = []
                # traditional match
                idx1, idx2 = 0, 0
                while idx1 < len(original_src_tokens) and idx2 < len(masked_src_tokens):
                    if original_src_tokens[idx1] == masked_src_tokens[idx2]:
                        idx1 += 1
                        idx2 += 1
                    else:
                        assert masked_src_tokens[idx2] == self.tokenizer.mask_token_id, f"source_tokens: {original_src_tokens}\nmasked_tokens: {masked_src_tokens}\nsource: {self.tokenizer.decode(original_src_tokens)}\n masked: {self.tokenizer.decode(masked_src_tokens)}\n"
                        # notice that the last token would not be [MASK] because the last edit label will be set to "$KEEP" 
                        diff_start = idx1
                        diff_end = idx1
                        while original_src_tokens[diff_end] != masked_src_tokens[idx2+1]:
                            diff_end += 1
                        # check if next token is equal or masked
                        # Raise Error when complicated situation encountered.
                        if diff_start == diff_end:
                            generation_edits.append(('insert', diff_start, diff_end, idx2, idx2+1))
                        else:
                            generation_edits.append(('replace', diff_start, diff_end, idx2, idx2+1))
                        idx2 += 1
                        idx1 = diff_end
                assert idx1 == len(original_src_tokens) and idx2 == len(masked_src_tokens), f"source_tokens: {original_src_tokens}\nmasked_tokens: {masked_src_tokens}\n auto_edits: {generation_edits}"
                assert len(generation_edits) == len(mask_positions), f"source_tokens: {original_src_tokens}\nmasked_tokens: {masked_src_tokens}\n auto_edits: {generation_edits}"
            
            # convert to detecion labels
            basic_label = [self.keep_label] * len(original_src_tokens)
            for edit in generation_edits:
                edit_type, start, end, _, _ = edit
                if edit_type == 'insert':
                    basic_label[end-1] = self.insert_label
                else:
                    for idx in range(start, end):
                        basic_label[idx] = self.error_label
            
            # use detection predictions as augment source to construct parallel training data
            src_tokens, tgt_tokens, edits = self.edit_extractor.limited_split_sentence(src, tgt, max_sentence_length)
            edit_labels = self.edit_extractor.edit_labels(edits=edits, src_tokens=src_tokens, tgt_tokens=tgt_tokens)
            example = self.from_detection_results_to_augmented_glm_example(basic_label[:len(src_tokens)], edit_labels, src_tokens, tgt_tokens)
            train_example = self.add_detection_prefix(example, src_tokens=src_tokens, edit_labels=edit_labels)
        except Exception as e:
            ERROR_NUM += 1
            logger.info(f"({ERROR_NUM})ERROR Occured while converting to training data using masked text.")
            train_example = self.convert_gec_sentence_pair_to_example(src=src, tgt=tgt, max_sentence_length=max_sentence_length)
        assert train_example['input_ids'].dtype == train_example['input_ids'].dtype == train_example['detection_labels'].dtype == train_example['position_ids'].dtype == int
        assert type(train_example['source_length']) == type(train_example['prefix_length']) == type(train_example['prefix_prompt_length']) == int
        # TODO: example check in masked_words mode
        return train_example

    
    def convert_gec_sentence_pair_to_example(self, src: str, tgt: str, max_sentence_length: int = 100):
        # src_tokens, tgt_tokens = self.edit_extractor.split_sentence(src), self.edit_extractor.split_sentence(tgt)
        # edits = self.edit_extractor.edit(src_tokens, tgt_tokens)
        src_tokens, tgt_tokens, edits = self.edit_extractor.limited_split_sentence(src, tgt, max_sentence_length)
        edit_labels = self.edit_extractor.edit_labels(edits=edits, src_tokens=src_tokens, tgt_tokens=tgt_tokens)
        example = self.from_edit_to_glm_example(edits=edits, original_src_tokens=src_tokens, original_tgt_tokens=tgt_tokens)
        train_example = self.add_detection_prefix(example, src_tokens=src_tokens, edit_labels=edit_labels)
        assert train_example['input_ids'].dtype == train_example['input_ids'].dtype == train_example['detection_labels'].dtype == train_example['position_ids'].dtype == int
        assert type(train_example['source_length']) == type(train_example['prefix_length']) == type(train_example['prefix_prompt_length']) == int
        return train_example

    def convert_gec_sentence_pair_to_detection_example(self, src: str, tgt: str, max_sentence_length: int = 100):
        '''
        将句子对转换为纯检测器的训练样本
        '''
        # src_tokens, tgt_tokens = self.edit_extractor.split_sentence(src), self.edit_extractor.split_sentence(tgt)
        # edits = self.edit_extractor.edit(src_tokens, tgt_tokens)
        src_tokens, tgt_tokens, edits = self.edit_extractor.limited_split_sentence(src, tgt, max_sentence_length)
        edit_labels = self.edit_extractor.edit_labels(edits=edits, src_tokens=src_tokens, tgt_tokens=tgt_tokens)
        temp_example = {
            'input_ids': np.array([], dtype=int), 
            'target_ids': np.array([], dtype=int), 
            'position_ids': np.array([[],[]], dtype=int),
            'source_length': 0,
        }
        train_example = self.add_detection_prefix(temp_example, src_tokens=src_tokens, edit_labels=edit_labels)
        assert train_example['input_ids'].dtype == train_example['input_ids'].dtype == train_example['detection_labels'].dtype == train_example['position_ids'].dtype == int
        assert type(train_example['source_length']) == type(train_example['prefix_length']) == type(train_example['prefix_prompt_length']) == int
        return train_example
    
    def convert_sentence_to_detection_example(self, src: str, max_sentence_length: int, enable_warning = False):
        src_tokens = self.edit_extractor.split_sentence(src, max_sentence_length=max_sentence_length, enable_warning=enable_warning)
        temp_example = {
            'input_ids': np.array([], dtype=int), 
            'target_ids': np.array([], dtype=int), 
            'position_ids': np.array([[],[]], dtype=int),
            'source_length': 0,
        }
        infer_example = self.add_detection_prefix(temp_example, src_tokens=src_tokens, edit_labels=[self.keep_label]*len(src_tokens))
        return infer_example

    def convert_detected_sentence_to_infer_example(self, src_tokens: List[int], edit_label_ids: List[int]):
        edit_labels = [self.edit_label_id_map[item] for item in edit_label_ids]
        # src_tokens = self.edit_extractor.split_sentence(src)
        example = self.from_edit_label_to_glm_infer_example(edit_labels=edit_labels, src_tokens=src_tokens)
        infer_example = self.add_detection_prefix(example, src_tokens=src_tokens, edit_labels=edit_labels)
        return infer_example

    def convert_masked_sentence_to_infer_example(self, src: str, masked_text: Union[str, List[str]], max_sentence_length: int):
        '''
        根据检错模型输出的MASK文本转换出用于生成的样本
        '''
        # tokenize
        src_tokens = self.edit_extractor.split_sentence(src, max_sentence_length=max_sentence_length)
        # masked_src_tokens = self.edit_extractor.split_sentence(masked_text, max_sentence_length=max_sentence_length)
        if type(masked_text) == str:
            masked_src_tokens = self.edit_extractor.split_sentence(masked_text, max_sentence_length=max_sentence_length, enable_warning=True)
        elif type(masked_text) == list:
            masked_src_tokens = self.tokenizer.convert_tokens_to_ids(masked_text)[:max_sentence_length-2]
            # add special tokens
            if src_tokens[0] in self.tokenizer.all_special_ids:
                masked_src_tokens.insert(0, src_tokens[0])
            if src_tokens[-1] in self.tokenizer.all_special_ids:
                masked_src_tokens.append(src_tokens[-1])
        else:
            raise NotImplementedError()

        # find [MASK]
        source_tokens = masked_src_tokens
        mask_tokens_position = []
        for i, token in enumerate(source_tokens):
            if token == self.tokenizer.pad_token_id:
                mask_tokens_position.append(i)
        source_position_ids = [list(range(len(source_tokens))), [0]*len(source_tokens)]
        source_length = len(source_tokens)

        # if theres a mask, add a <sop> token
        if mask_tokens_position:
            source_tokens = np.concatenate((source_tokens, [self.tokenizer.sop_token_id]), dtype=int)
            # loss_masks = np.ones(len(tokens), dtype=np.long)
            # loss_masks[:source_length] = 0
            source_position_ids = np.concatenate((source_position_ids, [[mask_tokens_position[0]], [1]]), axis=1, dtype=int)
            
        example = {
            'input_ids': np.array(source_tokens, dtype=int), 
            'target_ids': np.array([self._loss_ignore_id] * len(source_tokens), dtype=int), 
            'position_ids': source_position_ids,
            'source_length': source_length,
        }
        edit_labels = [self.keep_label]*len(src_tokens)
        infer_example = self.add_detection_prefix(example, src_tokens=src_tokens, edit_labels=edit_labels)
        # TODO: check example in masked words mode.
        return infer_example


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("../models/glm-large-chinese", trust_remote_code=True)
    model_config = AutoConfig.from_pretrained("../models/glm-large-chinese", trust_remote_code=True)

    # Inference
    inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a [MASK] in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    print(inputs)

    src = "虽然我对服装事业方面较谋生，但我在原公司的时候，工作成绩也不错，可以说能干的人。"
    tgt = "虽然我对服装方面较陌生，但我在原公司的时候，工作成绩还不错，可以说是一个能干的人。"

    class Config:
        text_cut=15
        num_labels=2
        prompt='请改正下面的句子中的语法错误：'
    config = Config()

    dataprocess = GLMDataProcessor(tokenizer=tokenizer, args=None, config=config)
    src_t, tgt_t, edits = dataprocess.edit_extractor.limited_split_sentence(src, tgt, config.text_cut)
    res = dataprocess.convert_gec_sentence_pair_to_example(src, tgt)
    print(res)

    src = "虽然很客气，但我并不会这样做。尤其是他是美丽的？"
    tgt = "我本人虽然很客气的人，但我并不会像他这样去做一件事。尤其是它是美丽的传说？"

    res = dataprocess.convert_gec_sentence_pair_to_example(src, tgt)
    print(res)

    src = "日前，网易、新浪等14家网站联合向全国互联网发出文明办网倡议书，号召营造健康文明的网络文化环境，清除不健康信息，已成为社会的共同呼唤、家长的强烈要求和保障未成年人的迫切需要。"
    tgt = "日前，网易、新浪等14家网站联合向全国互联网发出文明办网倡议书，号召营造健康文明的网络文化环境，清除不健康信息，这个倡议书已成为社会的共同呼唤、家长的强烈要求和保障未成年人的迫切需要。"
    res = dataprocess.convert_gec_sentence_pair_to_example(src, tgt)
    print(res)
