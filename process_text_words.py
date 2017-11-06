import json
import os
import string
import time
from collections import Counter
from nltk.tokenize import word_tokenize

print(os.getcwd())
dataset = 'BookCorpus'

vocab_size = 20000

min_len = 4
max_len = 500
len_saving_intervals = 5

lim_unk = 0.0


files = []

if dataset == 'WikipediaWestbury':

    files = [os.path.join(dataset, f) for f in os.listdir(dataset) if f.startswith('WestburyLab.Wikipedia.Corpus_')]

elif dataset == 'BookCorpus':

    files = [os.path.join('./data/', dataset, f) for f in os.listdir(os.path.join('./data/', dataset)) if f.startswith('books_large_p')]

elif dataset == 'Test':

    files = ['Test/test.txt']


print('first pass (computing word frequencies)')

start = time.clock()
num_sentences_processed = 0

word_counts = Counter()

for filename in files:

    with open(filename, 'r') as f:

        raw_text = f.readlines()

        word_counts.update(word_tokenize(' '.join(raw_text).lower()))

        print('file processed; time taken = ' + str(time.clock() - start) + ' seconds')

print('word frequencies computed; time taken = ' + str(time.clock() - start) + ' seconds')

del word_counts['-end.of.document']

valid_vocab = list(dict(word_counts.most_common(vocab_size)).keys())

if lim_unk != 0.0:
    unk_token = '<UNK>'
    valid_vocab.append(unk_token)

eos_token = '<EOS>'
valid_vocab.append(eos_token)

valid_vocab_index = {valid_vocab[i]: i for i in range(len(valid_vocab))}


out_dir = dataset + '/processed_vocab' + str(vocab_size) + '_' + str(min_len) + 'to' + str(max_len)

if lim_unk is not None:
    out_dir += '_limunk' + str(lim_unk)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(os.path.join(out_dir, 'valid_vocab.txt'), 'w') as out:
    
    d = json.dumps(valid_vocab)
    out.write(d)


print('second pass (tokenizing sentences)')

start = time.clock()
num_sentences_processed = 0

sentences = []

for filename in files:

    with open(filename, 'r') as f:

        for sentence in f:

            valid_sentence = True

            sentence = sentence.strip().lower()

            sentence_tokens = word_tokenize(sentence)

            if len(sentence_tokens) < min_len or len(sentence_tokens) > max_len:
                valid_sentence = False
                continue

            if sentence == '---END.OF.DOCUMENT---' or ' ' not in sentence:
                valid_sentence = False
                continue

            sentence_indexed = []

            for t in sentence_tokens:

                try:
                    sentence_indexed.append(valid_vocab_index[t])
                except KeyError:
                    if lim_unk == 0.0:
                        valid_sentence = False
                        break
                    else:
                        sentence_indexed.append(valid_vocab_index[unk_token])

            if lim_unk is not None and lim_unk > 0.0 and sentence_indexed.count(valid_vocab_index[unk_token]) / len(sentence_indexed) >= lim_unk:
                valid_sentence = False
                continue

            sentence_indexed.append(valid_vocab_index[eos_token])

            if valid_sentence:
                sentences.append(sentence_indexed)

            num_sentences_processed += 1

            if num_sentences_processed % 100000 == 0:
                print(str(num_sentences_processed) + ' sentences processed; time taken = ' + str(time.clock() - start) +
                      ' seconds')

print('There are ' + str(len(sentences)) + ' sentences.')


for lower in range(1, max_len, len_saving_intervals):

    upper = lower + len_saving_intervals - 1

    print('saving ' + str(lower) + ' to ' + str(upper))

    sentences_range = [s for s in sentences if lower <= len(s) <= upper]

    with open(os.path.join(out_dir, str(lower) + '-' + str(upper) + '.txt'), 'w') as index:

        d = json.dumps(sentences_range)
        index.write(d)
