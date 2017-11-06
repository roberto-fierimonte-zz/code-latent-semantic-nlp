import os
import re

from collections import Counter
from nltk import word_tokenize, sent_tokenize


def process_dataset(input_folder, output_folder):

    word_count = Counter()
    sent_counter = 0

    for f in os.listdir(input_folder):
        for line in open(os.path.join(input_folder, f), 'r'):
            tokens = word_tokenize(line)
            if len(tokens) <= 40:
                sent_counter += 1
                for t in tokens:
                    word_count[t] += 1

    top_words = {k for (k, _) in word_count.most_common(20000)}

    with open(os.path.join(output_folder, 'processed'), 'a') as out:
        for f in os.listdir(input_folder):
            for line in open(os.path.join(input_folder, f), 'r'):
                tokens = word_tokenize(line)
                if len(tokens) <= 40:
                    sent_counter += 1
                    out.write()



if __name__ == "__main__":
    input_folder = os.path.join(os.getcwd(), './data/')
    output_folder = input_folder

    process_dataset(input_folder, output_folder)