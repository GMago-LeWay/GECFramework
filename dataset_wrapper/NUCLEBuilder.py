import logging
import os
from copy import deepcopy

import datasets


_CITATION = """\
@inproceedings{dahlmeier-etal-2013-building,
    title = "Building a Large Annotated Corpus of Learner {E}nglish: The {NUS} Corpus of Learner {E}nglish",
    author = "Dahlmeier, Daniel  and
      Ng, Hwee Tou  and
      Wu, Siew Mei",
    booktitle = "Proceedings of the Eighth Workshop on Innovative Use of {NLP} for Building Educational Applications",
    month = jun,
    year = "2013",
    url = "https://aclanthology.org/W13-1703",
    pages = "22--31",
}
"""

_DESCRIPTION = """\
The National University of Singapore Corpus of Learner English (NUCLE) consists of 1,400 essays written by mainly Asian undergraduate students at the National University of Singapore
"""


_HOMEPAGE = "https://www.comp.nus.edu.sg/~nlp/corpora.html"

_LICENSE = "other"

_URLS = {
    "dummy_link": "https://example.com/"
}


class NUCLE(datasets.GeneratorBasedBuilder):
    """NUCLE dataset for grammatical error correction"""

    VERSION = datasets.Version("3.3.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="public", version=VERSION, description="Dummy public config so that datasets tests pass"),
        datasets.BuilderConfig(name="private", version=VERSION, description="Actual config used for loading the data")
    ]

    DEFAULT_CONFIG_NAME = "public"

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
        file_path = f"dummy.m2"
        if self.config.name == "private":
            data_dir = dl_manager.manual_dir
            if data_dir is not None:
                file_path = os.path.join(data_dir, "bea2019", "nucle.train.gold.bea19.m2")
            else:
                logging.warning("Manual data_dir not provided, so the data will not be loaded")
        else:
            logging.warning("The default config 'public' is intended to enable passing the tests and loading the "
                            "private data separately. If you have access to the data, please use the config 'private' "
                            "and provide the directory to the BEA19-formatted data as `data_dir=...`")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": file_path}
            )
        ]

    def _generate_examples(self, file_path):
        if not os.path.exists(file_path):
            return

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