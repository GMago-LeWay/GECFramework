import os
from copy import deepcopy

import datasets


_CITATION = """\
@inproceedings{yannakoudakis-etal-2011-new,
    title = "A New Dataset and Method for Automatically Grading {ESOL} Texts",
    author = "Yannakoudakis, Helen  and
      Briscoe, Ted  and
      Medlock, Ben",
    booktitle = "Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2011",
    url = "https://aclanthology.org/P11-1019",
    pages = "180--189",
}
"""

_DESCRIPTION = """\
The CLC FCE Dataset is a set of 1,244 exam scripts written by candidates sitting the Cambridge ESOL First Certificate 
in English (FCE) examination in 2000 and 2001. The dataset exposes the sentence-level pre-tokenized M2 version, totaling 
33236 sentences.
"""

_HOMEPAGE = ""

_LICENSE = "Custom, allowed for non-commercial research and educational purposes"

_URLS = {
    "clc_fce_bea19": "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz"
}


class CLCFCE(datasets.GeneratorBasedBuilder):
    """Cambridge Learner Corpus: First Certificate in English"""

    VERSION = datasets.Version("2.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("int32"),
                "src_tokens": datasets.Sequence(datasets.Value("string")),
                "tgt_tokens": datasets.Sequence(datasets.Value("string")),
                "text": datasets.Value("string"),
                "label": datasets.Value("string"),
                "corrections": [{
                    "idx_src": datasets.Sequence(datasets.Value("int32")),
                    "idx_tgt": datasets.Sequence(datasets.Value("int32")),
                    "corr_type": datasets.Value("string")
                }]
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS["clc_fce_bea19"]
        data_dir = dl_manager.manual_dir
        if data_dir is None:
            data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": os.path.join(data_dir, "fce", "m2", "fce.train.gold.bea19.m2")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"file_path": os.path.join(data_dir, "fce", "m2", "fce.dev.gold.bea19.m2")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"file_path": os.path.join(data_dir, "fce", "m2", "fce.test.gold.bea19.m2")},
            ),
        ]

    def _generate_examples(self, file_path):
        skip_edits = {"noop", "UNK", "Um"}
        with open(file_path, "r", encoding="utf-8") as f:
            idx_ex = 0
            src_sent, tgt_sent, corrections, offset = None, None, [], 0
            for idx_line, _line in enumerate(f):
                line = _line.strip()

                if len(line) > 0:
                    prefix, remainder = line[0], line[2:]
                    if prefix == "S":
                        src_sent = remainder.split(" ")
                        tgt_sent = deepcopy(src_sent)

                    elif prefix == "A":
                        annotation_data = remainder.split("|||")
                        idx_start, idx_end = map(int, annotation_data[0].split(" "))
                        edit_type, edit_text = annotation_data[1], annotation_data[2]
                        if edit_type in skip_edits:
                            continue

                        formatted_correction = {
                            "idx_src": list(range(idx_start, idx_end)),
                            "idx_tgt": [],
                            "corr_type": edit_type
                        }
                        annotator_id = int(annotation_data[-1])
                        assert annotator_id == 0

                        removal = len(edit_text) == 0 or edit_text == "-NONE-"
                        if removal:
                            for idx_to_remove in range(idx_start, idx_end):
                                del tgt_sent[offset + idx_to_remove]
                                offset -= 1

                        else:  # replacement/insertion
                            edit_tokens = edit_text.split(" ")
                            len_diff = len(edit_tokens) - (idx_end - idx_start)

                            formatted_correction["idx_tgt"] = list(
                                range(offset + idx_start, offset + idx_end + len_diff))
                            tgt_sent[offset + idx_start: offset + idx_end] = edit_tokens
                            offset += len_diff

                        corrections.append(formatted_correction)

                else:  # empty line, indicating end of example
                    yield idx_ex, {
                        "id": idx_ex,
                        "src_tokens": src_sent,
                        "tgt_tokens": tgt_sent,
                        "text": ' '.join(src_sent),
                        "label": ' '.join(tgt_sent),
                        "corrections": corrections
                    }
                    src_sent, tgt_sent, corrections, offset = None, None, [], 0
                    idx_ex += 1
