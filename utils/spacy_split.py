import spacy
import os
import argparse
import codecs

import en_core_web_sm
model = en_core_web_sm.load()

def retokenize(input_file, output_file):
    assert os.path.exists(input_file) and os.path.exists(os.path.dirname(output_file)), "please check input file and output directory."
    original_data = codecs.open(input_file, 'r', 'utf-8').readlines()
    with codecs.open(output_file, 'w', 'utf-8') as f:
        for line in original_data:
            en_doc = model(line.strip())
            tokenized = ' '.join([token.text for token in en_doc])
            f.write(tokenized + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='', help='input txt lines')
    parser.add_argument('-o', type=str, default='', help='output txt lines')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    retokenize(args.i, args.o)
