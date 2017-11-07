import os
import re
import numpy as np
import time

from collections import Counter
from nltk import word_tokenize, sent_tokenize


def tf_idf(word, sentence, sentences):

    tf = sentence.count(word) / len(sentence)

    n_sentences = sum(1 for sentence in sentences if word in sentence)

    idf = np.log(len(sentences) / (1 + n_sentences))

    return tf*idf


def process_dataset(input_folder, output_folder, vocab_size=20000):

    start = time.time()
    word_counter = Counter()
    sent_counter = 0
    file_counter = 0

    for f in os.listdir(input_folder):
        file_counter += 1
        for line in open(os.path.join(input_folder, f), 'r'):
            sent_counter += 1
            if (sent_counter % 1000000 == 0) and (sent_counter > 0):
                print("Read %dM sentences" % (sent_counter / 1000000))
            tokens = word_tokenize(line)
            for t in tokens + ['<EOS>']:
                word_counter[t] += 1

        print("Parsed %d files in %.2f minutes." % (file_counter, (time.time() - start) / 60.0))

    print("Parsed %d sentences in %.2f minutes." % (sent_counter, (time.time() - start) / 60.0))

    top_words = {k for (k, _) in word_counter.most_common(vocab_size)}

    with open(os.path.join(output_folder, 'processed.txt'), 'a') as out:
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