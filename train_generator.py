#! /usr/bin/env python
# encoding:utf-8

import time
import collections
import numpy as np
import tensorflow as tf
import prodata # load the dict
import loaddata # deal with the data which is used to train the generative model
import os
import generator
from tensorflow.python.ops import variable_scope

"""
pre-train the generator
"""


START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode',default_value ='train', docstring = 'run mode, if not train, the trained model will be used to generate sample')
tf.app.flags.DEFINE_string('model_path',default_value ='model_gen', docstring = 'Path expression to save the model .')
tf.app.flags.DEFINE_string('data_path',default_value ='data/taobao_traindata.txt', docstring = 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path',default_value ='data/dict.txt',docstring = 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_integer('vocab_size',default_value= 3000,docstring = 'number of vocab')
tf.app.flags.DEFINE_integer('latent_size',default_value= 60,docstring = 'the dim of the latent variable')
tf.app.flags.DEFINE_integer('embed_size',default_value= 200,docstring = 'dim of embedding')
tf.app.flags.DEFINE_integer('batch_size',default_value= 64,docstring = 'batch_size')
tf.app.flags.DEFINE_integer('hidden_size',default_value= 100,docstring = 'number of vocab')
tf.app.flags.DEFINE_integer('num_layers',default_value= 2,docstring = 'number of vocab')
tf.app.flags.DEFINE_integer('output_size',default_value= 200,docstring = 'number of vocab')
tf.app.flags.DEFINE_float('learning_rate',default_value=0.001,docstring='learning_rate')
tf.app.flags.DEFINE_integer('seq_length', default_value= 21, docstring = 'max length of sequence include start_token or end_token x,y,mask length')
tf.app.flags.DEFINE_integer('epoch', default_value= 10, docstring = 'the numbers of epoch')
tf.app.flags.DEFINE_integer('num_gen', default_value= 1000, docstring = 'the number of comment generated')
tf.app.flags.DEFINE_float('alpha',default_value=0.5,docstring=' alpha !!!!!!!! ')
tf.app.flags.DEFINE_float('dropout',default_value=0.5,docstring=' dropout !!!!!!!! ')
tf.app.flags.DEFINE_string('gpu',default_value ='1', docstring = 'which gpu')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

# max_sequence_length



def main(unused_argv):

    data_file = FLAGS.data_path
    vocab_path = FLAGS.vocab_path
    vocab_size = FLAGS.vocab_size
    vocab = prodata.Vocab(vocab_path,vocab_size)
    data = loaddata.Data_loader(FLAGS.batch_size,FLAGS.seq_length)
    vocab_freqs = data.create_batches(data_file,vocab)

    if FLAGS.mode == 'train':
        dropout = FLAGS.dropout
    else:
        dropout = 0

    batch_size = FLAGS.batch_size
    num_layers = FLAGS.num_layers
    learning_rate = FLAGS.learning_rate
    latent_size = FLAGS.latent_size

    Generator = generator.Generator(vocab_size, batch_size, emb_dim = FLAGS.embed_size, hidden_dim = FLAGS.hidden_size,
                     sequence_length=FLAGS.seq_length,start_token = START_TOKEN, end_token = END_TOKEN,
                    pad_token = PAD_TOKEN,outputsize = FLAGS.output_size,vocab_freqs = vocab_freqs,latent_size = latent_size,dropout=dropout,
                     num_layers = num_layers ,learning_rate=learning_rate, reward_gamma=0.95,scope='generator')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=Generator.g_params,max_to_keep=100)

    if FLAGS.mode == 'train':
        # train the generator and save the model after training

        for i in xrange(FLAGS.epoch):

            data.reset_pointer()

            for j in xrange(data.num_batch):

                x,y,mask = data.next_batch()

                feed = {Generator.x:x,
                        Generator.y:y,
                        Generator.mask:mask
                }

                loss,kl_loss,_ = sess.run([Generator.pre_loss,Generator.pre_loss2,Generator.pretrain_updates],
                                  feed)
                print 'epoch:%d,batch:%d,loss:%f,kl_loss:%f'%(i,j,loss,kl_loss)
            saver.save(sess, save_path=FLAGS.model_path + '/gen', global_step=i)

    else:
        # generate the examples by using the trained model

        model_file = tf.train.latest_checkpoint(FLAGS.model_path + '/.')
        saver.restore(sess, model_file)

        num_epoch = FLAGS.num_gen // batch_size

        for _ in xrange(num_epoch):

            x,y,mask = Generator.generate_samples(sess)
            for i in xrange(batch_size):
                comment_batch = []
                for j in xrange(FLAGS.seq_length):
                    if y[i, j] == END_TOKEN:
                        break
                    comment_batch.append(vocab.id2word(y[i, j]))
                buffer = ''.join(comment_batch)
                buffer_str = buffer.encode('utf-8')
                print buffer_str


if __name__ == '__main__':
  tf.app.run()


