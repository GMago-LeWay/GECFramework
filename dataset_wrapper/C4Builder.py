# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""


import csv
import glob
import os

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:c4_200m_dataset,
title = {c4_200m},
author={Li Liwei},
year={2021}
}
"""

_DESCRIPTION = """\
GEC Dataset Generated from C4
"""

_HOMEPAGE = "https://www.kaggle.com/a0155991rliwei/c4-200m"

_LICENSE = ""

_URL = "data.zip"


class C4200M(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "train"
    
    def _info(self):
        features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.manual_dir
        if data_dir is None:
            data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(
        self, filepath
    ):
        """ Yields examples as (key, example) tuples. """
        def fix_nulls(s):
            for line in s:
                yield line.replace('\0', ' ')

        path = os.path.join(filepath, "*.tsv*")
        for filename in glob.glob(path):
            with open(filename, encoding="utf-8") as f:
                reader = csv.reader(fix_nulls(f), delimiter="\t", quoting=csv.QUOTE_NONE)
                for id_, row in enumerate(reader):
                    yield id_, {
                        "text": row[0],
                        "label": row[1],
                    }
