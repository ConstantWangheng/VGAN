#! /usr/bin/env python
# encoding:utf-8



"""
The adversarsial training file

"""
import tensorflow as tf
import prodata
from load_data import Dis_dataloader
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
        print " generated samples in the text !!!!!!! "
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
tf.app.flags.DEFINE_string('data_test',default_value ='data/taobao_testdata.txt',docstring = 'Path expression to test data ')

tf.app.flags.DEFINE_string('vocab_path',default_value ='data/dict.txt',docstring = 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_integer('batch_size',default_value= 64,docstring = 'batch_size')
tf.app.flags.DEFINE_integer('num_gen', default_value= 20000, docstring = 'the number of comment generated')
tf.app.flags.DEFINE_integer('vocab_size',default_value= 3000,docstring = 'number of vocab')
tf.app.flags.DEFINE_integer('embed_size',default_value= 200,docstring = 'dim of embedding')
tf.app.flags.DEFINE_integer('hidden_size',default_value= 100,docstring = 'RNN hidden_size')
tf.app.flags.DEFINE_integer('output_size',default_value= 200,docstring = 'output size')
tf.app.flags.DEFINE_integer('num_layers',default_value= 2,docstring = 'gen layers')
tf.app.flags.DEFINE_integer('seq_length', default_value= 21, docstring = 'max length of sequence include start_token or end_token ----x,y,mask length')

tf.app.flags.DEFINE_string('pre_gen_model', default_value='model_gen', docstring = 'the pre_trained generator dir')
tf.app.flags.DEFINE_string('pre_dis_model', default_value= 'dis_pre_model', docstring = 'the pre_trained discriminator dir')
tf.app.flags.DEFINE_string('gen_model', default_value='model_1', docstring = 'the directory in which the generator will be saved during adversarisal training')
tf.app.flags.DEFINE_string('dis_model', default_value= 'model_2', docstring = 'the directory in which the discriminator will be saved during adversarisal training')

tf.app.flags.DEFINE_integer('total_epoch', default_value= 300, docstring = 'the adversial number of epoch')
tf.app.flags.DEFINE_integer('gen_epoch', default_value= 5, docstring = 'the training numbers of generator every epoch')
tf.app.flags.DEFINE_integer('dis_epoch', default_value= 3, docstring = 'the training numbers of discriminator every epoch')

tf.app.flags.DEFINE_integer('latent_size', default_value= 60, docstring = 'the latent size')
tf.app.flags.DEFINE_string('gpu', default_value= '2', docstring = 'the latent size')

tf.app.flags.DEFINE_float('reward_gamma',default_value = 0.95, docstring = 'reward ')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu



def main(x):
    
    import pdb;pdb.set_trace()
    pre_gen_model = FLAGS.pre_gen_model + '/.'
    # the pretrained generator

    pre_dis_model = FLAGS.pre_dis_model + '/.'
    # the pretrained discriminator

    gen_model_path = FLAGS.gen_model + '/gen'
    dis_model_path = FLAGS.dis_model + '/dis'

    vocab_path = FLAGS.vocab_path

    batch_size = FLAGS.batch_size

    data_neg_new = FLAGS.data_negative

    data_pos = FLAGS.data_positive

    vocab_size = FLAGS.vocab_size

    seq_length = FLAGS.seq_length

    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

    embed_dim = FLAGS.embed_size

    vocab1 = prodata.Vocab(vocab_path,vocab_size)

    data = loaddata.Data_loader(FLAGS.batch_size, FLAGS.seq_length)

    vocab_freqs = data.create_batches(data_pos, vocab1)

    vocab2 = prodata.Vocab(vocab_path,vocab_size)

    dis_data = Dis_dataloader(FLAGS.batch_size,FLAGS.seq_length)

    test_data = FLAGS.data_test

    test_data_nll = loaddata.Data_loader(FLAGS.batch_size, FLAGS.seq_length)
    test_data_nll.create_batches(test_data, vocab1)

    # the generator
    gen_model = Generator(vocab_size=vocab_size, batch_size=batch_size, emb_dim=embed_dim, hidden_dim=FLAGS.hidden_size,
                                    sequence_length=FLAGS.seq_length, start_token=START_TOKEN, end_token=END_TOKEN,
                                    pad_token=PAD_TOKEN, outputsize=FLAGS.output_size, vocab_freqs=vocab_freqs,
                                    latent_size=FLAGS.latent_size, dropout=0,
                                    num_layers=2, learning_rate=0.001, reward_gamma=0.95, scope='generator')

    # the policy notwork for reward
    gen_rollout = Generator(vocab_size=vocab_size, batch_size=batch_size, emb_dim=embed_dim, hidden_dim=FLAGS.hidden_size,
                                    sequence_length=FLAGS.seq_length, start_token=START_TOKEN, end_token=END_TOKEN,
                                    pad_token=PAD_TOKEN, outputsize=FLAGS.output_size, vocab_freqs=vocab_freqs,
                                    latent_size=FLAGS.latent_size, dropout=0,
                                    num_layers=2, learning_rate=0.001, reward_gamma=0.95, scope='rollout')

    # the discriminator
    dis_model = Discriminator(seq_length, num_classes=2, vocab_size=vocab_size,
                                                embedding_size=64, filter_sizes=dis_filter_sizes,
                                                num_filters=dis_num_filters, l2_reg_lambda=0.2)


    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    pdb.set_trace()
    saver1 = tf.train.Saver(var_list=gen_model.g_params,max_to_keep=500)

    model_file1 = tf.train.latest_checkpoint(pre_gen_model)

    saver3 = tf.train.Saver(var_list=gen_model.g_params,max_to_keep=500)

    saver1.restore(sess,model_file1)


    saver2 = tf.train.Saver(var_list=dis_model.params,max_to_keep=500)

    model_file2 = tf.train.latest_checkpoint(pre_dis_model)

    saver2.restore(sess,model_file2)

    saver4 = tf.train.Saver(var_list=dis_model.params,max_to_keep=500)

    gen_rollout.F_update_para(sess,gen_model.g_params)

    print ' start the adversial training !!!!!!!! '

    for i in xrange(FLAGS.total_epoch):
        for it in xrange(FLAGS.gen_epoch):
            samples_x,samples_y,mask = gen_model.generate_samples(sess)

            # get the reward by the policy network
            reward = get_reward(sess,samples_x,samples_y,seq_length,3,gen_rollout,dis_model,vocab1)
            # reward1.py give the reward of the sentence
            # Montle Carlo search is used

            # print samples_y
            # print reward

            feed = {gen_model.rewards: reward,
                    gen_model.mask: mask,
                    gen_model.x: samples_x,
                    gen_model.y: samples_y}

            # policy gradient update the generator
            loss1, _ = sess.run([gen_model.g_loss,
                                 gen_model.g_updates], feed)

            print '\n'
            print 'the batch_size samples generated !!!!!!'
            print 'the adversial epoch:',i
            print '\n'
            _, samples_batch, _ = gen_model.generate_samples(sess)

            for x_i in xrange(batch_size):
                comment_batch = []
                for x_j in xrange(seq_length):
                    if samples_batch[x_i, x_j] == END_TOKEN:
                        break
                    comment_batch.append(vocab1.id2word(samples_batch[x_i, x_j]))
                buffer = u'adversial:' + ''.join(comment_batch)
                buffer_str = buffer.encode('utf-8')
                print 'epoch:',i,'---',buffer_str
            print '\n'


            # calulate the nll of test dataset
            test_data_nll.reset_pointer()
            aveloss = 0.0
            for j in xrange(test_data_nll.num_batch):
                x, y, mask = test_data_nll.next_batch()

                feed = {gen_model.x: x,
                        gen_model.y: y,
                        gen_model.mask: mask
                        }

                loss_1 = sess.run(gen_model.pre_loss_sen,
                                  feed)
                loss_1 = loss_1 / float(batch_size)
                aveloss = aveloss + loss_1 / float(test_data_nll.num_batch)

            print 'epoch %d, batch %d, nll: %f' % (i, it, aveloss)


        print ' updates the rollout parameters !!!!!!!! '
        gen_rollout.update_para(sess,gen_model.g_params)

        print ' generate new examples file !!!!!!!!! '
        saver3.save(sess,gen_model_path,i)

        generate_samples(sess,gen_model,batch_size,FLAGS.num_gen,vocab1,data_neg_new)

        dis_data.load_train_data(data_pos, data_neg_new, vocab2)
        dis_data.reset_pointer()

        print '\n'
        print ' train the discriminator!!!!!!! '
        for _ in xrange(FLAGS.dis_epoch):
            for m1 in xrange(dis_data.num_batch):
                x,y = dis_data.next_batch()
                feed1 = {
                    dis_model.input_x: x,
                    dis_model.input_y: y,
                    dis_model.dropout_keep_prob: 0.75
                }
                loss2, _ = sess.run([dis_model.loss, dis_model.train_op], feed1)
                print 'epoch:%d,batch:%d,loss:%f'%(i,m1,loss2)
        print '\n'

        saver4.save(sess, save_path=dis_model_path, global_step=i)


if __name__ == '__main__':
  tf.app.run()



