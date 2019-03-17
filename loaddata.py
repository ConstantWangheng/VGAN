#! /usr/bin/env python
# encoding:utf-8

"""
load data for the generator
"""
import codecs
import numpy as np


class Data_loader():
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.token_stream = []
        self.token_stream_length = []
        self.sequence_length = sequence_length
        self.START_TOKEN = 0
        self.END_TOKEN = 1
        self.UNK_TOKEN = 2
        self.PAD_TOKEN = 3
        self.START_TOKEN_1 = u'start'
        self.END_TOKEN_1 = u'end'
        self.UNK_TOKEN_1 = u'unk'
        self.PAD_TOKEN_1 = u'pad'

    def create_batches(self, data_file,vocab):

        vocab_freds = np.zeros([vocab.count]) # count the word frequencies

        self.token_stream = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():

                vocab_freds[0] += 1

                vocab_freds[1] += 1

                for x in line:
                    vocab_freds[vocab.word2id(x)] += 1

                line = line.strip()
                line = line.split()
                parse_line = [self.START_TOKEN] + [vocab.word2id(x) for x in line] + [self.END_TOKEN]

                if len(parse_line) < self.sequence_length + 1: # 21 + 1

                    self.token_stream.append(parse_line +[self.PAD_TOKEN]*(self.sequence_length + 1 - len(parse_line)))
                    self.token_stream_length.append(len(parse_line)-1)
                    vocab_freds[3] += self.sequence_length + 1 - len(parse_line)

                else:

                    self.token_stream.append(parse_line[:self.sequence_length] + [self.END_TOKEN])
                    self.token_stream_length.append(self.sequence_length)




        self.num_batch = int(len(self.token_stream) / self.batch_size)

        self.mask = np.zeros((self.num_batch * self.batch_size, self.sequence_length),dtype=np.int32)


        for i in xrange(self.num_batch * self.batch_size):
            for j in xrange(self.sequence_length):
                if j < self.token_stream_length[i]:
                    self.mask[i, j] = 1
                else:
                    self.mask[i, j] = 0


        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.masks = np.split(self.mask, self.num_batch, 0)
        self.pointer = 0

        return vocab_freds

    def next_batch(self):
        ret = self.sequence_batch[self.pointer][:,:-1],self.sequence_batch[self.pointer][:,1:], self.masks[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret  # x,y,mask

    def reset_pointer(self):
        self.pointer = 0

