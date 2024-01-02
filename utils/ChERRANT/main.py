from utils.ChERRANT.parallel_to_m2 import to_m2
from utils.ChERRANT.compare_m2_for_evaluation import compare_m2


def compute_cherrant(ids, src_texts: list[str], tgt_texts: list[list[str]], predict_texts: list[str], device):
    total_sample_num = len(src_texts)
    assert total_sample_num == len(tgt_texts) == len(predict_texts)
    hyp_m2 = to_m2(ids, src_tgt_texts=[[src_texts[i]] + [predict_texts[i]] for i in range(total_sample_num)], device=device)
    ref_m2 = to_m2(ids, src_tgt_texts=[[src_texts[i]] + tgt_texts[i] for i in range(total_sample_num)], device=device)
    res = compare_m2(hyp_m2=hyp_m2, ref_m2=ref_m2)
    return res

def compute_cherrant_with_ref_file(ids, src_texts: list[str], predict_texts: list[str], ref_file: str, device):
    ref_m2 = open(ref_file).read().strip().split("\n\n")
    total_sample_num = len(src_texts)
    hyp_m2 = to_m2(ids, src_tgt_texts=[[src_texts[i]] + [predict_texts[i]] for i in range(total_sample_num)], device=device)
    res = compare_m2(hyp_m2=hyp_m2, ref_m2=ref_m2)
    return res
