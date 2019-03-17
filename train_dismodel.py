#! /usr/bin/env python
# encoding:utf-8



"""
pre-train the discriminator !!!

"""
import tensorflow as tf
import prodata
from load_data import Dis_dataloader
# pre deal with the data which is used to train the discriminator

from reward1 import get_reward
import random
import numpy as np
from generator import Generator
from discriminator1 import Discriminator
import time
import numpy as np
import loaddata
import os
import codecs

seed = 123
START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3


def generate_samples(sess,generator,batch_size,generated_num,vocab,output_file):
    """
    trainable_model:  generator
    batch_size: the number of sentences generated every time
    generated_num: the total number of sentences generated
    output_file: the generated examples are saved in the file

    """
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        samples = sess.run(generator.gen_x) # batch_size * seq_length
        generated_samples.extend(samples.tolist())

    generated_samples_noend = []

    for line in generated_samples:
        list1 = []
        for word in line[:-1]:
            if word == END_TOKEN or word == PAD_TOKEN:
                break
            else:
                list1.append(word)
        if len(list1) == 0:
            continue
        generated_samples_noend.append(list1)

    linelist = []
    for x in generated_samples_noend:
        line = []
        for x1 in x:
            line.append(vocab.id2word(x1))
        linelist.append(line)
    with codecs.open(output_file, 'w',encoding='utf-8') as fout:
        print '\n'
        print "generated samples in the text !!!!!!! "
        for i,line in enumerate(linelist):
            try:
                buffer = ' '.join(line) + u'\n'
                buffer_str = buffer.encode('utf-8')
                if i < 10:
                    print buffer_str
                fout.write(buffer)
            except:
                print 'some errors happen'
                continue
        print '\n'



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode',default_value ='train', docstring = 'train or decode train：train the adversial model, decode just generate sample ')
tf.app.flags.DEFINE_string('data_positive',default_value ='data/taobao_traindata.txt',docstring = 'Path expression to positive file')
tf.app.flags.DEFINE_string('data_negative',default_value ='data/data_neg.txt',docstring = 'Path expression to example generated ')
tf.app.flags.DEFINE_string('vocab_path',default_value ='data/dict.txt',docstring = 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('dis_path',default_value ='pre_dis_model',docstring = 'Path expression to dis model saved')
tf.app.flags.DEFINE_string('gen_path',default_value ='model_gen',docstring = 'Path expression to gen model saved')
tf.app.flags.DEFINE_integer('batch_size',default_value= 64,docstring = 'batch_size')
tf.app.flags.DEFINE_integer('num_gen', default_value= 20000, docstring = 'the number of comment generated')
tf.app.flags.DEFINE_integer('vocab_size',default_value= 3000,docstring = 'number of vocab')
tf.app.flags.DEFINE_integer('embed_size',default_value= 200,docstring = 'dim of embedding')
tf.app.flags.DEFINE_integer('hidden_size',default_value= 100,docstring = 'RNN hidden_size')
tf.app.flags.DEFINE_integer('output_size',default_value= 200,docstring = 'output size')
tf.app.flags.DEFINE_integer('num_layers',default_value= 2,docstring = 'RNN layers of RNN in the generator')
tf.app.flags.DEFINE_integer('seq_length', default_value= 21, docstring = 'max length of sequence include start_token or end_token ----x,y,mask length')
tf.app.flags.DEFINE_integer('total_epoch', default_value= 300, docstring = 'the adversial number of epoch')
tf.app.flags.DEFINE_float('dis_l2_reg_lambda',default_value = 0.2, docstring = 'dis l2 前面的一个超参数')
tf.app.flags.DEFINE_integer('latent_size', default_value= 60, docstring = 'the latent size')
tf.app.flags.DEFINE_string('gpu', default_value= '2', docstring = 'the latent size')
tf.app.flags.DEFINE_float('dropout',default_value = 0.5, docstring = 'dropout')

tf.app.flags.DEFINE_float('reward_gamma',default_value = 0.95, docstring = 'reward的一个比例参数')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu



def main(x):
    # the pretrained generator
    gen_model_path = FLAGS.gen_path + '/.'


    dis_model_path = FLAGS.dis_path + '/dis'

    vocab_path = FLAGS.vocab_path

    batch_size = FLAGS.batch_size

    data_neg_new = FLAGS.data_negative

    data_pos = FLAGS.data_positive

    seq_length = FLAGS.seq_length

    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


    vocab = prodata.Vocab(vocab_path,3000)

    data = loaddata.Data_loader(FLAGS.batch_size, FLAGS.seq_length)

    vocab_freqs = data.create_batches(data_pos, vocab)



    dis_data = Dis_dataloader(batch_size,FLAGS.seq_length)

    gen_model = Generator(vocab_size=3000, batch_size=200, emb_dim=200, hidden_dim=FLAGS.hidden_size,
                                    sequence_length=FLAGS.seq_length, start_token=START_TOKEN, end_token=END_TOKEN,
                                    pad_token=PAD_TOKEN, outputsize=FLAGS.output_size, vocab_freqs=vocab_freqs,
                                     dropout=0,
                                    num_layers=2, learning_rate=0.001, reward_gamma=0.95, scope='generator')


    dis_model = Discriminator(seq_length, num_classes=2, vocab_size=3000,
                                                embedding_size=64, filter_sizes=dis_filter_sizes,
                                                num_filters=dis_num_filters, l2_reg_lambda=0.2)


    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    saver1 = tf.train.Saver(var_list=gen_model.g_params,max_to_keep=100)

    model_file1 = tf.train.latest_checkpoint(gen_model_path)
    # load the pretrained generator

    saver1.restore(sess,model_file1)


    saver2 = tf.train.Saver(var_list=dis_model.params,max_to_keep=500)

    print ' start training the discriminator !!!!!!!! '



    for i in xrange(50):
        print ' train the discriminator!!!!!!! '
        generate_samples(sess, gen_model, 200, FLAGS.num_gen, vocab, data_neg_new)
        # generate the examples by using the pretrained generator
        # and examples are saved in data_neg_new

        dis_data.load_train_data(data_pos, data_neg_new, vocab)
        dis_data.reset_pointer()
        for m1 in xrange(dis_data.num_batch):
            x,y = dis_data.next_batch()
            feed1 = {
                dis_model.input_x: x,
                dis_model.input_y: y,
                dis_model.dropout_keep_prob: 0.75
            }
            loss2, _ = sess.run([dis_model.loss, dis_model.train_op], feed1)
            print 'epoch:%d,batch:%d,loss:%f'%(i,m1,loss2)

        saver2.save(sess, save_path=dis_model_path, global_step=i)


if __name__ == '__main__':
  tf.app.run()
