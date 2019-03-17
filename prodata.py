#! /usr/bin/env python
# encoding:utf-8

"""
build the vocabulary for the dict file

"""
import codecs

START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

START_TOKEN_1 = u'start'
END_TOKEN_1 = u'end'
UNK_TOKEN_1 = u'unk'
PAD_TOKEN_1 = u'pad'

class Vocab(object):
    def __init__(self,datafile,maxsize = -1):

        self.maxsize = maxsize
        self.vocablist = []
        self.vocab = {}
        self.count = 0  # 字典的大小
        self.vocab[START_TOKEN] = self.count
        self.vocablist.append(u'start')
        self.count += 1
        self.vocab[END_TOKEN] = self.count
        self.vocablist.append(u'end')
        self.count += 1
        self.vocab[UNK_TOKEN] = self.count
        self.vocablist.append(u'unk')
        self.count += 1
        self.vocab[PAD_TOKEN] = self.count
        self.vocablist.append(u'pad')
        self.count += 1

        with codecs.open(datafile,mode='r',encoding='utf-8') as f1:
            for line in f1.readlines():
                line = line.strip().split()
                word = line[0]
                if maxsize != -1 :
                    if self.count < maxsize:
                        self.vocab[word] = self.count
                        self.vocablist.append(word)
                        self.count += 1
                    else:
                        break
                else:
                    self.vocab[word] = self.count
                    self.vocablist.append(word)
                    self.count += 1

    def word2id(self,word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab[UNK_TOKEN]



    def id2word(self,id):

        return self.vocablist[id]

