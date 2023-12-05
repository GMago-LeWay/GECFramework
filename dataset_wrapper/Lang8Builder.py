import os
from copy import deepcopy

import datasets


_CITATION = """\
@inproceedings{rothe2021a,
  title = {{A Simple Recipe for Multilingual Grammatical Error Correction}},
  author = {Rothe, Sascha and Mallinson, Jonathan and Malmi, Eric and Krause, Sebastian and Severyn, Aliaksei},
  booktitle = {Proc. of ACL-IJCNLP},
  year = {2021}
}
"""

_DESCRIPTION = """\
Lang-8 is an online language learning website which encourages users to correct each other's grammar. The Lang-8 Corpus of Learner English is a somewhat-clean, English subsection of this website (Mizumoto et al., 2011; Tajiri et al., 2012).
"""

_HOMEPAGE = "https://www.cl.cam.ac.uk/research/nl/bea2019st/#data"

_LICENSE = "other"


class Lang8Builder(datasets.GeneratorBasedBuilder):
    """Lang-8 Corpus of Learner English"""

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
        data_dir = dl_manager.manual_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": os.path.join(data_dir, "lang8.train.auto.bea19.m2")},
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

                        # for current, annotator id > 0 will not be considered as gold eddit, emitted.
                        annotator_id = int(annotation_data[-1])
                        if annotator_id > 0:
                            continue
                        
                        assert annotator_id == 0
                        idx_start, idx_end = map(int, annotation_data[0].split(" "))
                        edit_type, edit_text = annotation_data[1], annotation_data[2]
                        if edit_type in skip_edits:
                            continue

                        formatted_correction = {
                            "idx_src": list(range(idx_start, idx_end)),
                            "idx_tgt": [],
                            "corr_type": edit_type
                        }

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
