#coding:utf-8

"""
calulate the reward for the generated sentence

"""

import tensorflow as tf

import numpy as np

START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

START_TOKEN_1 = u'start'
END_TOKEN_1 = u'end'
UNK_TOKEN_1 = u'unk'
PAD_TOKEN_1 = u'pad'


def get_reward(sess, input_x,input_y,seq_length,ave_num,generator,discriminator,vocab):


    rewards = []  # batch_size * seq_length
    batch_size = generator.batch_size

    for i in xrange(ave_num):

        for given_num in xrange(1, seq_length + 1):

            dataout = generator.gen_sample_give_num1(sess, input_x, input_y, given_num)



            feed = {discriminator.input_x: dataout, discriminator.dropout_keep_prob: 1.0}

            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)

            ypred = np.array([item[1] for item in ypred_for_auc])


            if i == 0:
                rewards.append(ypred)
            else:
                rewards[given_num - 1] += ypred


    rewards = np.transpose(np.array(rewards)) / (1.0 * ave_num)

    return rewards







