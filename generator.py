#coding:utf-8

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np


START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

def KLd(mu1, sig1, mu2, sig2, keep_dims=0):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.

    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    if keep_dims:
        kl = 0.5 * (2 * tf.log(sig2) - 2 * tf.log(sig1) +
                    (sig1 ** 2 + (mu1 - mu2) ** 2) / sig2 ** 2 - 1)
    else:
        kl = tf.reduce_sum(0.5 * (2 * tf.log(sig2) - 2 * tf.log(sig1) +
                          (sig1 ** 2 + (mu1 - mu2) ** 2) /
                          sig2 ** 2 - 1), axis=-1)

    return kl

def norm_embed(emb,vocab_freqs):
    # normalize the word vector
    weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev

class Generator(object):

    def __init__(self, vocab_size, batch_size, emb_dim, hidden_dim,
                 sequence_length,start_token, end_token, pad_token,outputsize,vocab_freqs,
                 latent_size = 60,dropout=0,
                 num_layers = 2 ,learning_rate=0.001, reward_gamma=0.95,scope='generator'):
        """
        vocab_size : the size of vocabulary
        hidden_dim : the hidden dim of RNN cell
        sequence_length : the max length of a sentence
        start_token : the start token of a sentence
        end_token : the end token of a sentence
        pad_token : the pad token of a sentence
        vocab_freqs : The number of words appearing in the vocab
        latent size : the dimension of the prior and posterior distribution
        num_layers : the RNN layers numbers
        """
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma

        self.num_layers = num_layers
        self.output_size = outputsize
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.latent_size = latent_size
        self.dropout = dropout
        self.vocab_freqs = vocab_freqs


        with tf.variable_scope(scope):
            self.g_embeddings = tf.get_variable(name='g_embeddings',shape=[self.vocab_size,self.emb_dim])
            with tf.variable_scope('layers'):
                cells = []

                for i in xrange(num_layers):
                    if i != 0:
                        cell = tf.contrib.rnn.ResidualWrapper(tf.nn.rnn_cell.GRUCell(hidden_dim))
                    else:
                        cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
                    cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=1 - self.dropout)
                    cells.append(cell)

                self.cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


            with tf.variable_scope('latent_distribution'):
                # calulate the prior distribution
                self.Wz_mean1 = tf.get_variable(
                    "Wz_mean1", [self.num_layers * self.hidden_dim, self.latent_size],
                    dtype=tf.float32)
                self.bz_mean1 = tf.get_variable(
                    "bz_mean1", [self.latent_size], dtype=tf.float32)

                self.Wz_var1 = tf.get_variable(
                    "Wz_var1", [self.num_layers * self.hidden_dim, self.latent_size],
                    dtype=tf.float32)
                self.bz_var1 = tf.get_variable(
                    "bz_var1", [self.latent_size], dtype=tf.float32)

                # calulate the posterior distribution
                self.Wz_mean2 = tf.get_variable(
                    "Wz_mean2", [self.num_layers * self.hidden_dim + self.emb_dim, self.latent_size],
                    dtype=tf.float32)
                self.bz_mean2 = tf.get_variable(
                    "bz_mean2", [self.latent_size], dtype=tf.float32)

                self.Wz_var2 = tf.get_variable(
                    "Wz_var2", [self.num_layers * self.hidden_dim + self.emb_dim, self.latent_size],
                    dtype=tf.float32)
                self.bz_var2 = tf.get_variable(
                    "bz_var2", [self.latent_size], dtype=tf.float32)


                self.W_z = tf.get_variable(
                    "W_z", [self.latent_size, self.output_size], dtype=tf.float32)
                self.b_z = tf.get_variable(
                    "b_z", [self.output_size], dtype=tf.float32)

            with tf.variable_scope('softmax_output'):
                self.softmax_w = tf.get_variable("softmax_w", [outputsize + hidden_dim , vocab_size])
                self.softmax_b = tf.get_variable("softmax_b", [vocab_size])

            self.state1 = self.cell.zero_state(batch_size, dtype=tf.float32)

            # placeholder definition
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
            self.y = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
            self.mask = tf.placeholder(tf.int32, shape=[self.batch_size,self.sequence_length])
            self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size,self.sequence_length])

            self.vocab_freqs = tf.convert_to_tensor(self.vocab_freqs, dtype=tf.float32)
            self.vocab_freqs = tf.expand_dims(self.vocab_freqs, axis=-1)

            with tf.device("/cpu:0"):

                self.g_embeddings = norm_embed(self.g_embeddings, self.vocab_freqs)
                self.processed_x = tf.nn.embedding_lookup(self.g_embeddings, self.x)
                # batch * seq_length * emb_dim

            self.probs,self.last_state1,z_prior,z_posterior = self.pre_train(self.processed_x,self.state1)


            kl_loss = []
            for z_p, z_po in zip(z_prior, z_posterior):
                kl_loss.append(KLd(z_p[0], z_p[1], z_po[0], z_po[1]))
            kl_loss = tf.stack(kl_loss)  # seq_length * batch
            kl_loss = tf.transpose(kl_loss, perm=[1, 0])  # batch * seq_length

            pre_total_nums = tf.cast(tf.reduce_sum(self.mask), dtype=tf.float32)

            self.pre_loss2 = tf.reduce_sum(kl_loss * tf.cast(self.mask, dtype=tf.float32)) / pre_total_nums

            pre_total_loss = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), vocab_size, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.probs, [-1, vocab_size]), 1e-20, 1.0)
                    ), 1) * tf.cast(tf.reshape(self.mask, [-1]), dtype=tf.float32))

            pre_total_loss_nomask = -tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), vocab_size, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.probs, [-1, vocab_size]), 1e-20, 1.0)
                    ))



            self.pre_loss1 = pre_total_loss / pre_total_nums

            self.pre_loss = self.pre_loss1 + 0.1 * self.pre_loss2




            self.pre_loss_sen = pre_total_loss_nomask

            # pretrain_opt = self.g_optimizer(self.learning_rate)

            self.g_params =[]

            for p1 in tf.trainable_variables():
                if scope in p1.name:
                    self.g_params.append(p1)

            # self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pre_loss, self.g_params), self.grad_clip)
            # self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

            grads, _ = tf.clip_by_global_norm(tf.gradients(self.pre_loss, self.g_params), 5)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.pretrain_updates = optimizer.apply_gradients(zip(grads, self.g_params))

            # ********************************************************************************
            # unsupervised
            # ********************************************************************************
            # with tf.variable_scope("reward_train"):


            self.g_loss1 = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.probs, [-1, self.vocab_size]), 1e-20, 1.0)
                    ), 1) * tf.cast(tf.reshape(self.mask, [-1]), dtype=tf.float32) * tf.reshape(self.rewards, [-1])
            )  # reward shape batch_size * seq_length

            self.g_loss = self.g_loss1 + 0.1 * self.pre_loss2 # pre_loss2 表示的是KL散度

            grads1, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
            optimizer1 = tf.train.AdamOptimizer(self.learning_rate)
            self.g_updates = optimizer1.apply_gradients(zip(grads1, self.g_params))

            # ********************************************************************************
            # generate some sentences by inputting the start token
            # ********************************************************************************

            self.gen_x = []
            cell_state = self.state1
            cell_input = tf.constant([self.start_token] * batch_size, dtype=tf.int32)

            for _ in xrange(self.sequence_length): # seq_length  21
                next_token , state_ = self.g_recurrent(cell_input,cell_state)
                cell_state = state_
                cell_input = next_token
                # the last output is as the next input
                self.gen_x.append(next_token)
            self.gen_x = tf.stack(self.gen_x) # shape seq_len * batch_size
            self.gen_x = tf.transpose(tf.reshape(self.gen_x,[self.sequence_length,batch_size]),perm=[1,0])
            #  batch * seq_length

            # ********************************************************************************
            # just run a timestep
            # *******************************************************************************

            self.x_token1 = tf.placeholder(dtype=tf.int32,shape=[batch_size])
            self.x_token1_state1 =  self.state1 #self.cell.zero_state(batch_size,dtype=tf.float32)
            self.x_token1_y ,self.x_token1_state1_ = self.pre_current(self.x_token1,self.x_token1_state1)

            # ********************************************************************************
            # just run a timestep
            # *******************************************************************************

            self.x_token2 = tf.placeholder(dtype=tf.int32, shape=[batch_size])
            self.x_token2_state1 = self.state1
            self.x_token2_y, self.x_token2_state1_ = self.g_recurrent(self.x_token2, self.x_token2_state1)

            # ********************************************************************************

    def pre_train(self,inputs,state):
        # batch * seq_length * hidden_size

        inputs_list = tf.unstack(inputs,axis=1)
        outputs_list = []
        z_prior = []
        z_posterior = []

        for _,x in enumerate(inputs_list):

            state_concat = tf.concat(state,axis=-1)
            z1_input = state_concat

            # the prior distribution
            z1_mu = tf.matmul(z1_input, self.Wz_mean1) + self.bz_mean1
            z1_stv = tf.nn.softplus(tf.matmul(z1_input, self.Wz_var1) + self.bz_var1)  # 标准差

            eps = tf.random_normal(tf.stack([self.batch_size, self.latent_size]))


            output1,state = self.cell(x,state)

            # the posterior distribution
            z2_input = tf.concat(state,axis=-1)


            z2_input = tf.concat([z2_input,x],axis=-1)

            z2_mu = tf.matmul(z2_input, self.Wz_mean2) + self.bz_mean2
            z2_stv = tf.nn.softplus(tf.matmul(z2_input, self.Wz_var2) + self.bz_var2)  # 标准差

            # get the sample from the posterior distribution
            z = tf.add(z2_mu, tf.multiply(z2_stv, eps))

            z = tf.nn.tanh(tf.matmul(z, self.W_z) + self.b_z)  # batch * hidden_dim

            z_prior.append((z1_mu,z1_stv))
            z_posterior.append((z2_mu,z2_stv))

            outputs_list.append(tf.concat([output1, z], axis=-1))



        last_state1 = state
        outputs = tf.stack(outputs_list)
        outputs = tf.transpose(outputs,perm=[1,0,2])
        output_in = tf.reshape(outputs, [-1, self.output_size + self.hidden_dim])
        logits = tf.matmul(output_in, self.softmax_w) + self.softmax_b

        probs = tf.nn.softmax(logits)

        return probs,last_state1,z_prior,z_posterior
        # probs [-1,vocab_size]
        #  last_state1 [batch_size,hidden_size]


    def g_recurrent(self,x_token,state):


        x = tf.nn.embedding_lookup(self.g_embeddings,x_token)

        y,state_ = self.cell(x,state)

        state_concat = tf.concat(state_, axis=-1)
        z2_input = tf.concat([state_concat, x], axis=1)

        z2_mu = tf.matmul(z2_input, self.Wz_mean2) + self.bz_mean2  # 均值
        z2_stv = tf.nn.softplus(tf.matmul(z2_input, self.Wz_var2) + self.bz_var2)  # 标准差
        eps = tf.random_normal(tf.stack([self.batch_size, self.latent_size]))

        z = tf.add(z2_mu, tf.multiply(z2_stv, eps))  # 潜变量得到的采样结果
        z = tf.nn.tanh(tf.matmul(z, self.W_z) + self.b_z)  # 形状是 batch * hidden_dim

        output_in = tf.concat([y, z], axis=1)
        logits = tf.matmul(output_in, self.softmax_w) + self.softmax_b

        probs = tf.nn.softmax(logits)  # batch * vocab_size

        log_probs = tf.log(probs)

        next_token = tf.cast(tf.reshape(tf.multinomial(log_probs, 1), [self.batch_size]), tf.int32)

        return next_token,state_

    def pre_current(self,x_token,state):

        x = tf.nn.embedding_lookup(self.g_embeddings, x_token)
        y, state_ = self.cell(x, state)

        state_concat = tf.concat(state_, axis=-1)
        z2_input = tf.concat([state_concat, x], axis=1)

        z2_mu = tf.matmul(z2_input, self.Wz_mean2) + self.bz_mean2
        z2_stv = tf.nn.softplus(tf.matmul(z2_input, self.Wz_var2) + self.bz_var2)  # 标准差

        eps = tf.random_normal(tf.stack([self.batch_size, self.latent_size]))

        z = tf.add(z2_mu, tf.multiply(z2_stv, eps))

        z = tf.nn.tanh(tf.matmul(z, self.W_z) + self.b_z)

        output_in = tf.concat([y, z], axis=1)


        logits = tf.matmul(output_in, self.softmax_w) + self.softmax_b
        probs = tf.nn.softmax(logits)  # batch * vocab_size

        y = tf.argmax(probs,dimension=1) # batch_size
        y = tf.reshape(y,[self.batch_size])
        y = tf.cast(y,dtype=tf.int32)
        return y , state_

    def generate_samples(self,sess):
        """
        generate samples
        """
        xmask = np.zeros(shape=[self.batch_size,self.sequence_length],dtype=np.int32)
        samples = sess.run(self.gen_x)
        dataout = np.ones(shape=[self.batch_size,self.sequence_length]) * PAD_TOKEN
        for i in xrange(self.batch_size):
            for j in xrange(self.sequence_length):
                if samples[i,j] == END_TOKEN or samples[i,j] == PAD_TOKEN:
                    dataout[i,j] = END_TOKEN
                    if j < self.sequence_length:
                        xmask[i,j] = 1
                    break
                else:
                    dataout[i,j] = samples[i,j]
                    xmask[i,j] = 1
        dataout_start_token = np.zeros([self.batch_size,1],dtype=np.int32)
        dataout = np.concatenate([dataout_start_token,dataout],axis=1)
        dataout = dataout.astype(int)
        return dataout[:,:-1],dataout[:,1:],xmask



    def gen_sample_give_num(self,sess,inputx,inputy,give_num):

        inputx_tensor = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        inputy_tensor = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        inputx_list = tf.unstack(inputx_tensor,axis=1)
        inputy_list = tf.unstack(inputy_tensor,axis=1)

        start_token = tf.constant([self.start_token] * self.batch_size, dtype=tf.int32)

        output_list = []

        state = []
        for _ in xrange(self.num_layers):
            state.append(tf.zeros(shape=[self.batch_size, self.hidden_dim]))
        state = tuple(state)


        seq_length = self.sequence_length # 总共循环这么多次
        for i in xrange(seq_length):
            if i < give_num:
                inputy_list[i] = tf.cast(inputy_list[i],dtype=tf.int32)
                output_list.append(inputy_list[i])
                cell_input = inputx_list[i]
                cell_in = tf.nn.embedding_lookup(self.g_embeddings,cell_input)
                cell_output,state_ = self.cell(cell_in,state)
                state = state_

            else:

                if i == 0:
                    cell_input = start_token
                else:
                    cell_input = output_list[-1]

                cell_in = tf.nn.embedding_lookup(self.g_embeddings,cell_input)
                cell_output, state_ = self.cell(cell_in, state)


                state_concat = tf.concat(state_, axis=-1)
                z2_input = tf.concat([state_concat, cell_in], axis=1)
                z2_mu = tf.matmul(z2_input, self.Wz_mean2) + self.bz_mean2
                z2_stv = tf.nn.softplus(tf.matmul(z2_input, self.Wz_var2) + self.bz_var2)
                eps = tf.random_normal(tf.stack([self.batch_size, self.latent_size]))
                z = tf.add(z2_mu, tf.multiply(z2_stv, eps))
                z = tf.nn.tanh(tf.matmul(z, self.W_z) + self.b_z)


                output_in = tf.concat([cell_output, z], axis=1)
                logits = tf.matmul(output_in, self.softmax_w) + self.softmax_b
                probs = tf.nn.softmax(logits)  # batch * vocab_size
                log_probs = tf.log(probs)
                next_token = tf.cast(tf.reshape(tf.multinomial(log_probs, 1), [self.batch_size]), tf.int32)
                # next_token [batch_size]

                state = state_
                output_list.append(next_token)

        # output_list[0] = tf.cast(output_list[0],dtype=tf.int32)
        output_tensor = tf.stack(output_list)
        output_given = tf.transpose(output_tensor,perm=[1,0])

        output_given = sess.run(output_given,{inputx_tensor:inputx,
                                              inputy_tensor:inputy})

        dataout = np.ones(shape=[self.batch_size, self.sequence_length]) * PAD_TOKEN
        for i in xrange(self.batch_size):
            for j in xrange(self.sequence_length):
                if output_given[i, j] == END_TOKEN or output_given[i, j] == PAD_TOKEN:
                    dataout[i, j] = END_TOKEN
                    break
                else:
                    dataout[i, j] = output_given[i, j]

        dataout = dataout.astype(int)

        return dataout

    def gen_sample_give_num1(self, sess, inputx, inputy, give_num):

        inputx_list = []
        inputy_list = []
        start_token = np.zeros([self.batch_size],dtype=int)

        for i in xrange(self.sequence_length):
            inputx_list.append(inputx[:,i])
            inputy_list.append(inputy[:,i])

        outputy_list = []
        state = sess.run(self.state1)

        for i in xrange(self.sequence_length):
            if i < give_num:

                input_1 = inputx_list[i]
                y_,state_ = sess.run([self.x_token1_y,self.x_token1_state1_],{
                    self.x_token1:input_1,
                    self.x_token1_state1:state})
                outputy_list.append(inputy_list[i])
                state = state_

            else:
                if i == 0:
                    input_1 = start_token
                else:
                    input_1 = outputy_list[-1]

                y_,state_ = sess.run([self.x_token2_y,self.x_token2_state1_],{
                    self.x_token2:input_1,
                    self.x_token2_state1:state})
                state = state_
                outputy_list.append(y_)

        outputy_arr = np.array(outputy_list)  # seq_length * batch_size

        outputy_arr = np.transpose(outputy_arr)

        dataout = np.ones(shape=[self.batch_size, self.sequence_length]) * PAD_TOKEN
        for i in xrange(self.batch_size):
            for j in xrange(self.sequence_length):
                if outputy_arr[i, j] == END_TOKEN or outputy_arr[i, j] == PAD_TOKEN:
                    dataout[i, j] = END_TOKEN
                    break
                else:
                    dataout[i, j] = outputy_arr[i, j]

        dataout = dataout.astype(int)
        return dataout

    def F_update_para(self,sess,params):


        for x,y in zip(self.g_params,params):

            sess.run(tf.assign(x,tf.identity(y)))



    def update_para(self,sess,params,update_rate=0.8):

        for x,y in zip(self.g_params,params):
            if 'g_embeddings' in x.name and 'g_embeddings' in y.name:
                sess.run(tf.assign(x,tf.identity(y)))
                continue
            sess.run(tf.assign(x,(1-update_rate)*x + update_rate * tf.identity(y)))




