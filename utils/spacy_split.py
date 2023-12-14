import spacy
import os
import argparse
import codecs
import re

import en_core_web_sm
en = en_core_web_sm.load()

def bea_retokenize(input_file, output_file):
    assert os.path.exists(input_file) and os.path.exists(os.path.dirname(output_file)), "please check input file and output directory."
    original_data = codecs.open(input_file, 'r', 'utf-8').readlines()
    with codecs.open(output_file, 'w', 'utf-8') as f:
        for line in original_data:
            line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                          "'\\1", line)
            line = " ".join([t.text for t in en.tokenizer(line)])
            # in spaCy v1.9.0 and the en_core_web_sm-1.2.0 model
            # 80% -> 80%, but in newest ver. 2.3.9', 80% -> 80 %
            # haven't -> haven't, but in newest ver. 2.3.9', haven't -> have n't
            line = re.sub("(?<=\d)\s+%", "%", line)
            line = re.sub("((?:have)|(?:has)) n't", "\\1n't", line)
            line = re.sub("^-", "- ", line)
            line = re.sub(r"\s+", " ", line)
            f.write(line.strip() + '\n')

retokenization_rules = [
    # Remove extra space around single quotes, hyphens, and slashes.
    (" ' (.*?) ' ", " '\\1' "),
    (" ?- ?", "-"),
    (" / ", "/"),
    # Ensure there are spaces around parentheses and brackets.
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
]

def conll_retokenize(input_file, output_file):
    assert os.path.exists(input_file) and os.path.exists(os.path.dirname(output_file)), "please check input file and output directory."
    original_data = codecs.open(input_file, 'r', 'utf-8').readlines()
    with codecs.open(output_file, 'w', 'utf-8') as f:
        for line in original_data:
            line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                          "'\\1", line)
            line = " ".join([t.text for t in en.tokenizer(line)])
            for rule in retokenization_rules:
                line = re.sub(rule[0], rule[1], line)
            f.write(line.strip() + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='', help='dataset: bea/conll')
    parser.add_argument('-i', type=str, default='', help='input txt lines')
    parser.add_argument('-o', type=str, default='', help='output txt lines')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.d == 'bea':
        bea_retokenize(args.i, args.o)
    elif args.d == 'conll':
        conll_retokenize(args.i, args.o)
    else:
        raise NotImplementedError()
