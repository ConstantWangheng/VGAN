#coding:utf-8



"""
load data for discriminator

"""
import codecs
import numpy as np


class Dis_dataloader():
    def __init__(self, batch_size, sequence_length):


        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.sequence_length = sequence_length
        self.PAD_TOKEN = 3
        self.END_TOKEN = 1
        self.UNK_TOKEN = 2
        self.START_TOKEN = 0

    def load_train_data(self, positive_file, negative_file, vocab):
        # Load data
        positive_examples = []
        positive_examples_all = []
        negative_examples = []

        with codecs.open(positive_file, 'r', encoding='utf-8') as fin1:
            for line in fin1.readlines():
                line = line.strip()
                line = line.split()
                parse_line = [vocab.word2id(x) for x in line] + [self.END_TOKEN]

                length_line = len(parse_line)
                if length_line <= self.sequence_length :
                    parse_line = parse_line + [self.PAD_TOKEN] * (self.sequence_length - len(parse_line))
                else:
                    parse_line = parse_line[:self.sequence_length] + [self.END_TOKEN]
                positive_examples_all.append(parse_line)

        with codecs.open(negative_file, 'r', encoding='utf-8') as fin2:
            for line in fin2.readlines():
                line = line.strip()
                line = line.split()
                parse_line = [vocab.word2id(x) for x in line] + [self.END_TOKEN]
                length_line = len(parse_line)
                if length_line <= self.sequence_length :
                    parse_line = parse_line + [self.PAD_TOKEN] * (self.sequence_length - len(parse_line))

                else:
                    parse_line = parse_line[:self.sequence_length-1] + [self.END_TOKEN]

                negative_examples.append(parse_line)

        positive_examples_num = len(positive_examples_all)
        shuffle_indices_pos = np.random.permutation(np.arange(positive_examples_num))

        for neg_x in xrange(len(negative_examples)):
            positive_examples.append(positive_examples_all[shuffle_indices_pos[neg_x]])

        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels

        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

