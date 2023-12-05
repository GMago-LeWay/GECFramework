from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error
import torch
import os
import json
from difflib import SequenceMatcher
import logging
import evaluate

logger = logging.getLogger(__name__)

class Metrics():
    def __init__(self):
        self.metrics_dict = {
            'cls': self._eval_multi,
            'linear': self._eval_regression,
            'spelling_check_1': SpellingCheckMetric.metrics,
            'spelling_check_2': SpellingCheckMetricCPN.compute_prf,
        }
    
    def get_metrics(self, metric_type:str):
        return self.metrics_dict[metric_type]

    def _eval_binary(self, y_pred, y_true):
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        eval_results = {
            "acc": round(accuracy_score(y_true=y_true, y_pred=y_pred), 4),
            "recall": round(recall_score(y_true=y_true, y_pred=y_pred), 4),
            "f1": round(f1_score(y_true=y_true, y_pred=y_pred) , 4),
        }
        return eval_results

    def _eval_multi(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_pred probit -> 0,1
        y_pred = ( y_pred.detach() > 0.5 ).int()
        eval_results = {'acc_avg': 0., 'recall_avg': 0., 'f1_avg': 0.}
        items = y_pred.shape[1]
        for i in range(items):
            # dim i binary metrics
            results_i = self._eval_binary(y_pred=y_pred[:, i:i+1], y_true=y_true[:, i:i+1])
            eval_results['acc_avg'] += results_i['acc']
            eval_results['recall_avg'] += results_i['recall']
            eval_results['f1_avg'] += results_i['f1']
            for key in results_i:
                eval_results[f'{key}_{i}'] = results_i[key]

        eval_results['acc_avg'] /= items
        eval_results['recall_avg'] /= items
        eval_results['f1_avg'] /= items
        return eval_results

    def _eval_spelling_check_1(self, x: list[str], y_pred: list[str], y_true: list[str]):
        assert len(x) == len(y_pred) == len(y_true)
        samples_len = len(x)
        sentences = [[x[i], y_true[i], y_pred[i]] for i in range(samples_len)]
        return SpellingCheckMetricCPN.metrics(sentences)

    def _eval_spelling_check_2(self, x: list[str], y_pred: list[str], y_true: list[str]):
        assert len(x) == len(y_pred) == len(y_true)
        samples_len = len(x)
        sentences = [[x[i], y_true[i], y_pred[i]] for i in range(samples_len)]
        return SpellingCheckMetric.metrics(sentences)

    def _eval_regression(self, y_pred, y_true):
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        eval_results = {
            "mse": round(mean_squared_error(y_true=y_true, y_pred=y_pred), 4),
            "mse_last": round(mean_squared_error(y_true=y_true[:, -1:], y_pred=y_pred[:, -1:]), 4),
        }
        return eval_results        


class SpellingCheckMetric:
    def __init__(self) -> None:
        pass

    @staticmethod
    # char level
    def char_D(sentences, verbose=False):
        total = 0
        TP = 0  # 正确识别
        predict_positive = 0  # 算法识别为“有错误”
        fact_positive = 0  # 语料中所有错误
        k = 0
        for sentence in sentences:
            k += 1
            total += 1
            lines0 = sentence[0]
            lines1 = sentence[1]
            lines2 = sentence[2]
            assert len(lines0) == len(lines1) == len(lines2), \
                "Inconsistent text length in spelling check. %dth sentence:(%d / %d / %d). %s//%s//%s" % (k, len(lines0), len(lines1), len(lines2), lines0, lines1, lines2)
            
            lines_list = [[lines0[i], lines1[i], lines2[i]] for i in range(min(len(lines0), len(lines1), len(lines2)))]
            # lines1 = [char for char in lines[1]]
            # lines2 = [char for char in lines[2]]
            for char_list in lines_list:
                if char_list[0] != char_list[1] and char_list[0] != char_list[2]:
                    TP += 1
                if char_list[0] != char_list[1]:
                    fact_positive += 1
                if char_list[0] != char_list[2]:
                    predict_positive += 1

        #print(total, cor, al_wro, wro)
        precision = TP / predict_positive if predict_positive else 0
        recall = TP / fact_positive if fact_positive else 0
        f2 = precision * recall * 2 / (precision + recall) if precision + recall else 0
        #print(total, cor, al_wro, wro)

        if verbose:
            print("********************************")
            print("char_D的准确率：", precision, TP, predict_positive)
            print("char_D的召回率：", recall, TP, fact_positive)
            print("char_D的F值：", f2, precision * recall * 2, (precision + recall))
            print(str(precision*100) + "\t" + str(recall*100) + "\t" + str(f2*100))
            print("|{:.2f}({}/{})|{:.2f}({}/{})|{:.2f}".format(precision * 100, TP, predict_positive, recall * 100, TP, fact_positive, f2 * 100))

        return {"precision": precision, "recall": recall, "f": f2}


    @staticmethod
    def char_C(sentences, verbose=False):
        total = 0  
        TP = 0  
        FP = 0  # 非错别字被误报为错别字
        FN = 0 
        # al_wro = 0  
        # wro = 0 

        k = 0
        for sentence in sentences:
            k += 1
            total += 1
            lines0 = sentence[0]
            lines1 = sentence[1]
            lines2 = sentence[2]
            lines_list = [[lines0[i], lines1[i], lines2[i]] for i in range(min(len(lines0), len(lines1), len(lines2)))]
            # lines1 = [char for char in lines[1]]
            # lines2 = [char for char in lines[2]]
            for char_list in lines_list:
                if char_list[0] != char_list[1] and char_list[1] == char_list[2]:
                    TP += 1
                if char_list[2] != char_list[1] and char_list[0] != char_list[2]:
                    FP += 1
                if char_list[0] != char_list[1] and char_list[1] != char_list[2]:
                    FN += 1

        #print(TP, FP, FN)
        al_wro = TP + FP
        wro = TP + FN
        precision = TP / al_wro if al_wro else 0
        recall = TP / wro if wro else 0
        #print(total, TP, al_wro, wro)
        #print(precision, recall)
        f2 = precision * recall * 2 / (precision + recall) if precision + recall else 0

        if verbose:
            print("********************************")
            print("char_C的准确率：", precision, TP, al_wro)
            print("char_C的召回率：", recall, TP, wro)
            print("char_C的F值：", f2, precision * recall * 2, (precision + recall))
            print(str(precision*100) + "\t" + str(recall*100) + "\t" + str(f2*100))
            print("|{:.2f}({}/{})|{:.2f}({}/{})|{:.2f}".format(precision*100, TP, al_wro, recall*100, TP, wro, f2*100))
        return {"precision": precision, "recall": recall, "f1": f2}

    @staticmethod
    def sentence_correction(sentences, verbose=False):
        total = 0
        TP = 0   # sentences which have errors and are correctly revised.
        FP = 0   # sentences which do not have errors but are revised.
        FN = 0   # sentences which have errors but are not correctly revised.
        def sentence_equal(s1, s2):
            if len(s1) != len(s2):
                return False
            for i in range(len(s1)):
                if s1[i] != s2[i]:
                    return False
            return True

        for sentence in sentences:
            total += 1
            lines0 = sentence[0]
            lines1 = sentence[1]
            lines2 = sentence[2]
            assert len(lines0) == len(lines1) == len(lines2), \
                "Inconsistent text length in spelling check. %dth sentence:(%d / %d / %d). %s//%s//%s" % (total, len(lines0), len(lines1), len(lines2), lines0, lines1, lines2)
            if not sentence_equal(lines0, lines1) and sentence_equal(lines1, lines2):
                TP += 1
            if sentence_equal(lines0, lines1) and not sentence_equal(lines0, lines2):
                FP += 1
            if not sentence_equal(lines0, lines1) and not sentence_equal(lines1, lines2):
                FN += 1
        precision = TP/(TP+FP) if TP+FP else 0
        recall = TP/(TP+FN) if TP+FP else 0
        f1 = 2*precision*recall / (precision+recall) if precision+recall else 0
        return {"sentence_precision": precision, "sentence_recall": recall, "sentence_f1": f1}

    @staticmethod
    def metrics(sentences: list[list[int]], verbose=False):
        """
        sentence item: [text, label, model_output]
        """
        res_dict = {}
        res_dict_d = SpellingCheckMetric.char_D(sentences, verbose)
        res_dict_c = SpellingCheckMetric.char_C(sentences, verbose)
        for key in res_dict_d:
            res_dict["detection_"+key] = res_dict_d[key]
        for key in res_dict_c:
            res_dict["correction_"+key] = res_dict_c[key]    

        sentence_level_dict = SpellingCheckMetric.sentence_correction(sentences, verbose)
        res_dict = {**res_dict, **sentence_level_dict}
        return res_dict

class SpellingCheckMetricCPN:
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_prf(results, verbose=False):
        TP = 0
        FP = 0
        FN = 0
        all_predict_true_index = []
        all_gold_index = []
        for item in results:
            src, tgt, predict = item
            gold_index = []
            each_true_index = []
            for i in range(len(list(src))):
                if src[i] == tgt[i]:
                    continue
                else:
                    gold_index.append(i)
            all_gold_index.append(gold_index)
            predict_index = []
            for i in range(len(list(src))):
                if src[i] == predict[i]:
                    continue
                else:
                    predict_index.append(i)

            for i in predict_index:
                if i in gold_index:
                    TP += 1
                    each_true_index.append(i)
                else:
                    FP += 1
            for i in gold_index:
                if i in predict_index:
                    continue
                else:
                    FN += 1
            all_predict_true_index.append(each_true_index)

        # For the detection Precision, Recall and F1
        detection_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
        detection_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
        if verbose:
            print("The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall, detection_f1))

        TP = 0
        FP = 0
        FN = 0

        for i in range(len( all_predict_true_index)):
            # we only detect those correctly detected location, which is a different from the common metrics since
            # we wanna to see the precision improve by using the confusionset
            if len(all_predict_true_index[i]) > 0:
                predict_words = []
                for j in all_predict_true_index[i]:
                    predict_words.append(results[i][2][j])
                    if results[i][1][j] == results[i][2][j]:
                        TP += 1
                    else:
                        FP += 1
                for j in all_gold_index[i]:
                    if results[i][1][j]  in predict_words:
                        continue
                    else:
                        FN += 1

        # For the correction Precision, Recall and F1
        correction_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
        correction_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
        correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall) if (correction_precision + correction_recall) > 0 else 0
        if verbose:
            print("The correction  result is precision={}, recall={} and F1={}".format(correction_precision, correction_recall, correction_f1))

        return {"detection_precision": detection_precision, "detection_recall": detection_recall, "detection_f1": detection_f1,
                "correction_precision": correction_precision, "correction_recall": correction_recall, "correction_f1": correction_f1}


class GECMetric:
    def __init__(self) -> None:
        pass

    @staticmethod
    def f1(precision, recall):
        if precision + recall == 0:
            return 0
        return round(2 * precision * recall / (precision + recall), 4)
    
    @staticmethod
    def f_alpha(precision, recall, alpha=0.5):
        if precision + recall == 0:
            return 0
        return round((1+alpha**2) * precision * recall / ((alpha**2)*precision + recall), 4)
    
    @staticmethod
    def compute_label_nums(src_text, trg_text, pred_text, log_error_to_fp=None):
        assert len(src_text) == len(trg_text) == len(
            pred_text), 'src_text:{}, trg_text:{}, pred_text:{}'.format(src_text, trg_text, pred_text)
        pred_num, detect_num, correct_num, ref_num = 0, 0, 0, 0

        for j in range(len(trg_text)):
            src_char, trg_char, pred_char = src_text[j], trg_text[j], pred_text[j]
            if src_char != trg_char:
                ref_num += 1
                if src_char != pred_char:
                    detect_num += 1
                elif log_error_to_fp is not None and pred_char != trg_char and pred_char == src_char:
                    log_text = '漏报\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, src_char, trg_char, pred_char, j)
                    log_error_to_fp.write(log_text)

            if src_char != pred_char:
                pred_num += 1
                if pred_char == trg_char:
                    correct_num += 1
                elif log_error_to_fp is not None and pred_char != trg_char and src_char == trg_char:
                    log_text = '误报\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, src_char, trg_char, pred_char, j)
                    log_error_to_fp.write(log_text)
                elif log_error_to_fp is not None and pred_char != trg_char and src_char != trg_char:
                    log_text = '错报(检对报错)\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, src_char, trg_char, pred_char, j)
                    log_error_to_fp.write(log_text)

        return (pred_num, detect_num, correct_num, ref_num)
    
    @staticmethod
    def ctc_f1(src_texts, trg_texts, pred_texts, log_error_to_fp=None):
        """训练过程中字级别序列标注任务的F1计算
        Args:
            src_texts ([type]): [源文本]
            trg_texts ([type]): [目标文本]
            pred_texts ([type]): [预测文本]
            log_error_to_fp : 文本路径
        Returns:
            [type]: [description]
        """
        if isinstance(src_texts, str):
            src_texts = [src_texts]
        if isinstance(trg_texts, str):
            trg_texts = [trg_texts]
        if isinstance(pred_texts, str):
            pred_texts = [pred_texts]
        lines_length = len(trg_texts)
        assert len(src_texts) == lines_length == len(
            pred_texts), 'keep equal length'
        all_pred_num, all_detect_num, all_correct_num, all_ref_num = 0, 0, 0, 0
        if log_error_to_fp is not None:
            f = open(log_error_to_fp, 'w', encoding='utf-8')
            f.write('type\tsrc_text\ttrg_text\tsrc_char\ttrg_char\tpred_char\tchar_index\n')
        else:
            f = None

        all_nums = [GECMetric.compute_label_nums(src_texts[i], trg_texts[i], pred_texts[i], f)
                    for i in range(lines_length)]
        if log_error_to_fp is not None:
            f.close()
        for i in all_nums:
            all_pred_num += i[0]
            all_detect_num += i[1]
            all_correct_num += i[2]
            all_ref_num += i[3]

        d_precision = round(all_detect_num / all_pred_num,
                            4) if all_pred_num != 0 else 0
        d_recall = round(all_detect_num / all_ref_num, 4) if all_ref_num != 0 else 0
        c_precision = round(all_correct_num / all_pred_num,
                            4) if all_pred_num != 0 else 0
        c_recall = round(all_correct_num / all_ref_num, 4) if all_ref_num != 0 else 0

        d_f1, c_f1 = GECMetric.f1(d_precision, d_recall), GECMetric.f1(c_precision, c_recall)

        logger.info('====== [Char Level] ======')
        logger.info('d_precsion:{}%, d_recall:{}%, d_f1:{}%'.format(
            d_precision * 100, d_recall * 100, d_f1 * 100))
        logger.info('c_precsion:{}%, c_recall:{}%, c_f1:{}%'.format(
            c_precision * 100, c_recall * 100, c_f1 * 100))
        logger.info('error_char_num: {}'.format(all_ref_num))
        return (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1)
    
    @staticmethod
    def ctc_comp_f1_sentence_level(src_texts, pred_texts, trg_texts):
        "计算负样本的 句子级 纠正级别 F1"
        correct_ref_num, correct_pred_num, correct_recall_num, correct_f1 = 0, 0, 0, 0
        for src_text, pred_text, trg_text in zip(src_texts, pred_texts, trg_texts):
            if src_text != pred_text:
                correct_pred_num += 1
            if src_text != trg_text:
                correct_ref_num += 1
            if src_text != trg_text and pred_text == trg_text:
                correct_recall_num += 1

        # assert correct_ref_num > 0, '文本中未发现错误，无法计算指标，该指标只计算含有错误的样本。'
        if correct_ref_num == 0:
            return 1, 1, 1

        correct_precision = 0 if correct_recall_num == 0 else correct_recall_num / correct_pred_num
        correct_recall = 0 if correct_recall_num == 0 else correct_recall_num / correct_ref_num
        correct_f1 = GECMetric.f1(correct_precision, correct_recall)

        return correct_precision, correct_recall, correct_f1
    
    @staticmethod
    def ctc_comp_f1_token_level(src_texts, pred_texts, trg_texts):
        "字级别，负样本 检测级别*0.8+纠正级别*0.2 f1"

        def compute_detect_correct_label_list(src_text, trg_text):
            detect_ref_list, correct_ref_list = [], []
            diffs = SequenceMatcher(None, src_text, trg_text).get_opcodes()
            for (tag, src_i1, src_i2, trg_i1, trg_i2) in diffs:

                if tag == 'replace':
                    # assert src_i2 - src_i1 == trg_i2 - trg_i1
                    if src_i2 - src_i1 != trg_i2 - trg_i1:
                        # logger.info("Warning: Replace operates unequal segments while computing metric.")
                        if src_i2 - src_i1 > trg_i2 - trg_i1:
                            src_i2 = src_i1 + trg_i2 - trg_i1
                        else:
                            trg_i2 = trg_i1 + src_i2 - src_i1
                    for count, src_i in enumerate(range(src_i1, src_i2)):
                        trg_token = trg_text[trg_i1 + count]
                        detect_ref_list.append(src_i)
                        correct_ref_list.append((src_i, trg_token))

                elif tag == 'delete':
                    trg_token = 'D' * (src_i2 - src_i1)
                    detect_ref_list.append(src_i1)
                    correct_ref_list.append((src_i1, trg_token))

                elif tag == 'insert':
                    trg_token = trg_text[trg_i1:trg_i2]
                    detect_ref_list.append(src_i1)
                    correct_ref_list.append((src_i1, trg_token))

            return detect_ref_list, correct_ref_list

        # 字级别
        detect_ref_num, detect_pred_num, detect_recall_num, detect_f1 = 0, 0, 0, 0
        correct_ref_num, correct_pred_num, correct_recall_num, correct_f1 = 0, 0, 0, 0

        for src_text, pred_text, trg_text in zip(src_texts, pred_texts, trg_texts):
            # 先统计检测和纠正标签
            try:
                detect_ref_list, correct_ref_list = compute_detect_correct_label_list(
                    src_text, trg_text)
            except Exception as e:
                # 可能Eval dataset有个别错误，暂时跳过
                continue
            try:
                # 处理bad case
                detect_pred_list, correct_pred_list = compute_detect_correct_label_list(
                    src_text, pred_text)
            except Exception as e:
                logger.exception(e)
                detect_pred_list, correct_pred_list = [], []

            detect_ref_num += len(detect_ref_list)
            detect_pred_num += len(detect_pred_list)
            detect_recall_num += len(set(detect_ref_list)
                                    & set(detect_pred_list))

            correct_ref_num += len(correct_ref_list)
            correct_pred_num += len(correct_pred_list)
            correct_recall_num += len(set(correct_ref_list)
                                    & set(correct_pred_list))

        if correct_ref_num == 0:
            # '文本中未发现错误，无法计算指标，该指标只计算含有错误的样本。'
            return 1, 1, [1, 1, 1], [1, 1, 1]

        detect_precision = 0 if detect_pred_num == 0 else detect_recall_num / detect_pred_num
        detect_recall = 0 if detect_ref_num == 0 else detect_recall_num / detect_ref_num

        correct_precision = 0 if detect_pred_num == 0 else correct_recall_num / correct_pred_num
        correct_recall = 0 if detect_ref_num == 0 else correct_recall_num / correct_ref_num

        detect_f1 = GECMetric.f1(detect_precision, detect_recall)
        correct_f1 = GECMetric.f1(correct_precision, correct_recall)
        correct_f0_5 = GECMetric.f_alpha(correct_precision, correct_recall, 0.5)

        final_f1 = detect_f1 * 0.8 + correct_f1 * 0.2

        return final_f1, correct_f0_5, [detect_precision, detect_recall, detect_f1], [correct_precision, correct_recall, correct_f1]
    
    @staticmethod
    def final_f1_score(src_texts,
                    pred_texts,
                    trg_texts,
                    log_fp=None):
        """"最终输出结果F1计算，综合了句级F1和字级F1"
        Args:
            src_texts (_type_): 源文本
            pred_texts (_type_): 预测文本
            trg_texts (_type_): 目标文本
            log_fp (str, optional): _description_. Defaults to 'logs/f1_score.log'.
        Returns:
            _type_: _description_
        """

        token_level_f1, correct_f_0_5, detect_metrics, correct_metrcis = GECMetric.ctc_comp_f1_token_level(
            src_texts, pred_texts, trg_texts)
        sent_level_p, sent_level_r, sent_level_f1 = GECMetric.ctc_comp_f1_sentence_level(
            src_texts, pred_texts, trg_texts)
        final_f1 = round(0.8 * token_level_f1 + sent_level_f1 * 0.2, 4)

        json_data = {

            'token_level:[detect_precision, detect_recall, detect_f1]': detect_metrics,
            'token_level:[correct_precision, correct_recall, correct_f1] ': correct_metrcis,
            'token_level:f0.5': correct_f_0_5,
            'token_level:f1': token_level_f1,

            'sentence_level:[correct_precision, correct_recall]': [sent_level_p, sent_level_r],
            'sentence_level:f1': sent_level_f1,

            'final_f1': final_f1
        }
        if log_fp:
            _log_fp = open(log_fp, 'w', encoding='utf-8')
            json.dump(json_data, _log_fp, indent=4)
        # logger.info('final f1:{}'.format(final_f1))
        # logger.info('f1 logfile saved at:{}'.format(log_fp))
        return json_data
    
    @staticmethod
    def test():
        from pprint import pprint
        src_texts = ["生体不错。", "我看建设这种机器是很值得的事情，所有的学校已该建设。"]
        pred_texts = ["身体不错。", "我看建设这种机器是很值得的事情，所有的学校以该建设。"]
        trg_texts = ["身体不错。", "我看建设这种机器是很值得的事情，所有的学校应该建设。"]
        pprint(GECMetric.final_f1_score(src_texts, pred_texts, trg_texts, None))


class DetectionCorrectionMetrics:
    def __init__(self, model_type, labels_num=3, loss_ignore_id=-100) -> None:
        assert labels_num in [2,3]
        assert model_type in ['all', 'detection', 'generate']
        self.model_type = model_type
        self.labels_num = labels_num
        self.loss_ignore_id = loss_ignore_id
        self.accuracy_metric = evaluate.load("utils/accuracy")

    def accuracy(self, references, predictions, weights):
        if weights.sum() == 0:
            return 0
        else:
            return self.accuracy_metric.compute(references=references, predictions=predictions, sample_weight=weights)['accuracy']
        
    def glm_accuracy(self, glm_pred_ids, glm_labels):
        glm_pred_ids, glm_labels = glm_pred_ids.ravel(), glm_labels.ravel()
        glm_pred_weights = (1 - (glm_labels == -100)*1).ravel()
        glm_accuracy = self.accuracy(references=glm_labels, predictions=glm_pred_ids, weights=glm_pred_weights)
        return glm_accuracy
    
    def detection_metrics(self, detection_pred_ids, detection_labels):
        detection_pred_ids, detection_labels = detection_pred_ids.ravel(), detection_labels.ravel()
        detection_pred_weights = (1 - (detection_labels == -100)*1).ravel()
        keep_pred_weights = ((detection_labels==0)*1).ravel()
        error_pred_weights = ((detection_labels==1)*1).ravel()
        detection_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=detection_pred_weights)
        keep_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=keep_pred_weights)
        error_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=error_pred_weights)

        if self.labels_num == 3:
            insert_pred_weights = ((detection_labels==2)*1).ravel()
            insert_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=insert_pred_weights)
            detection_geometric_accuracy = ( keep_accuracy*error_accuracy*insert_accuracy ) ** (1/3)
            return {'detection_accuracy': detection_accuracy, 'detection_geometric_accuracy': detection_geometric_accuracy,
                    'keep_accuracy': keep_accuracy, 'error_accuracy': error_accuracy, 'insert_accuracy': insert_accuracy, 'error_acc_sum': error_accuracy+insert_accuracy}
        else:
            detection_geometric_accuracy = ( keep_accuracy*error_accuracy ) ** (1/2)
            return {'detection_accuracy': detection_accuracy, 'detection_geometric_accuracy': detection_geometric_accuracy,
                    'keep_accuracy': keep_accuracy, 'error_accuracy': error_accuracy, 'error_acc_sum': error_accuracy}
        
    def metrics(self, glm_pred_ids, glm_labels, detection_pred_ids, detection_labels):
        if self.model_type == 'generate':
            glm_accuracy = self.glm_accuracy(glm_pred_ids=glm_pred_ids, glm_labels=glm_labels)
            return {'general_accuracy': glm_accuracy, 'glm_accuracy': glm_accuracy}
        else:
            detection_metrics = self.detection_metrics(detection_pred_ids=detection_pred_ids, detection_labels=detection_labels)
            if self.model_type == 'detection':
                detection_metrics['general_accuracy'] = detection_metrics['detection_geometric_accuracy']
                return detection_metrics
            else:
                glm_accuracy = self.glm_accuracy(glm_pred_ids=glm_pred_ids, glm_labels=glm_labels)
                ## avg of glm acc and balanced detecion acc
                general_accuracy = (glm_accuracy*detection_metrics['detection_geometric_accuracy'])**0.5
                ## avg of glm acc and detecion acc
                geometric_accuracy = (glm_accuracy*detection_metrics['detection_accuracy'])**0.5
                metrics = {'general_accuracy': general_accuracy, 'geometric_accuracy': geometric_accuracy, 'glm_accuracy': glm_accuracy,}
                for key in detection_metrics:
                    metrics[key] = detection_metrics[key]
                ## metirc for aggressive detection
                metrics['ad_accuracy'] = metrics['error_acc_sum'] + glm_accuracy
                return metrics

    def metrics_func(self):
        def compute(eval_predictions):
            pred_ids, label_ids = eval_predictions.predictions, eval_predictions.label_ids
            glm_pred_ids, detection_pred_ids = None, None
            if 'glm' in pred_ids[0]:
                glm_pred_ids = pred_ids[0]['glm']
            if 'detection' in pred_ids[0]:
                detection_pred_ids = pred_ids[0]['detection']
            glm_labels, detection_labels = label_ids
            return self.metrics(glm_pred_ids=glm_pred_ids, glm_labels=glm_labels, detection_pred_ids=detection_pred_ids, detection_labels=detection_labels)
        return compute


class CHERRANT:
    def __init__(self) -> None:
        pass

    @staticmethod
    def f1(precision, recall):
        if precision + recall == 0:
            return 0
        return round(2 * precision * recall / (precision + recall), 4)
    
    @staticmethod
    def f_alpha(precision, recall, alpha=0.5):
        if precision + recall == 0:
            return 0
        return round((1+alpha**2) * precision * recall / ((alpha**2)*precision + recall), 4)


if __name__ == "__main__":
    # SpellingCheckMetricCPN.compute_prf([['我斯你的石么从', '我是你的什么人', '我是您的石么众']], verbose=True)
    # SpellingCheckMetric.char_C([['我斯你的石么从', '我是你的什么人', '我是您的石么众']], verbose=True)
    print(GECMetric.test())

