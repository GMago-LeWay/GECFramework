import os
from copy import deepcopy

import datasets


_CITATION = """\
@article{wi_locness,
author = {Helen Yannakoudakis and Øistein E Andersen and Ardeshir Geranpayeh and Ted Briscoe and Diane Nicholls},
title = {Developing an automated writing placement system for ESL learners},
journal = {Applied Measurement in Education},
volume = {31},
number = {3},
pages = {251-267},
year  = {2018},
doi = {10.1080/08957347.2018.1464447},
}
"""

_DESCRIPTION = """\
Write & Improve is an online web platform that assists non-native English students with their writing. Specifically, students from around the world submit letters, stories, articles and essays in response to various prompts, and the W&I system provides instant feedback. Since W&I went live in 2014, W&I annotators have manually annotated some of these submissions and assigned them a CEFR level.
The LOCNESS corpus consists of essays written by native English students. It was originally compiled by researchers at the Centre for English Corpus Linguistics at the University of Louvain. Since native English students also sometimes make mistakes, we asked the W&I annotators to annotate a subsection of LOCNESS so researchers can test the effectiveness of their systems on the full range of English levels and abilities.
"""

_HOMEPAGE = "https://www.cl.cam.ac.uk/research/nl/bea2019st/"

_LICENSE = "other"

_URLS = {
    "wi_locness": "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
}


class WILocness(datasets.GeneratorBasedBuilder):
    """Write&Improve and LOCNESS dataset for grammatical error correction. """

    VERSION = datasets.Version("2.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="A", version=VERSION, description="CEFR level A"),
        datasets.BuilderConfig(name="B", version=VERSION, description="CEFR level B"),
        datasets.BuilderConfig(name="C", version=VERSION, description="CEFR level C"),
        datasets.BuilderConfig(name="N", version=VERSION, description="Native essays from LOCNESS"),
        datasets.BuilderConfig(name="all", version=VERSION, description="All training and validation data combined")
    ]

    DEFAULT_CONFIG_NAME = "all"

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
        urls = _URLS["wi_locness"]
        data_dir = dl_manager.manual_dir
        # print("---------------------", data_dir)
        if data_dir is None:
            data_dir = dl_manager.download_and_extract(urls)
        if self.config.name in {"A", "B", "C"}:
            splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"file_path": os.path.join(data_dir, "wi+locness", "m2", f"{self.config.name}.train.gold.bea19.m2")},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"file_path": os.path.join(data_dir, "wi+locness", "m2", f"{self.config.name}.dev.gold.bea19.m2")},
                )
            ]
        elif self.config.name == "N":
            splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"file_path": os.path.join(data_dir, "wi+locness", "m2", "N.dev.gold.bea19.m2")},
                )
            ]
        else:
            splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"file_path": os.path.join(data_dir, "wi+locness", "m2", f"ABC.train.gold.bea19.m2")},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"file_path": os.path.join(data_dir, "wi+locness", "m2", f"ABCN.dev.gold.bea19.m2")},
                )
            ]

        return splits

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
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
