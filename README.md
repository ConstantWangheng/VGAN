# VGAN
The code for the paper "text generation based on generative adversarial nets with latent variable"

Requirements:
	Tensorflow 1.3.0
	Python 2.7


## About this code
This code is based on the research paper [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient]. 

[SeqGAN code](https://github.com/LantaoYu/SeqGAN)


## How to run

### the dataset
1、PTB dataset 
2、Taobao Reviews 
3、Amazon Food Reviews




### Run training

### First, pretrain the generator  --- train_generator.py
### Second, pretrain the discriminator --- train_dismodel.py
### last, adversarisal training --- train_gan.py



For example, taobao reviews
1、To pretrain the generator, run:
python train_generator.py --mode=train --model_path=model_gen --data_path=data/taobao_traindata.txt --vocab_path=data/dict.txt --vocab_size=3000 --latent_size=60 --embed_size=200 --batch_size=64 --hidden_size=100 --num_layers=2 --output_size=200 --learning_rate=0.001 --seq_length=21 --epoch=50 --num_gen=1000 --dropout=0.5 --gpu=0


the pretrained generator is saved in the directory 'model_gen'.


2、To pretrain the discriminator, run:
python train_dismodel.py

python train_dismodel.py --mode=train --data_positive=data/taobao_traindata.txt --data_negative=data/data_neg.txt --gen_path=model_gen --dis_path=pre_dis_model --num_gen=20000 --vocab_size=3000 

note: 
data_positive: the train data 
data_negative: the file in which the generated examples is saved. It is built during the discriminator is trained.
gen_path: the directory in which the pretrained generator is saved.

In the train_dismodel.py, the parameters of generator should be initialized via the parameters of pretrained generator.


3、adversarial training 
python train_gan.py --mode=train --data_positive=data/taobao_traindata.txt --data_negative=data/data_neg.txt --data_test=data/taobao_testdata.txt --vocab_path=data/dict.txt --batch_size=64 --num_gen=20000 --vocab_size=3000 --embed_size=200 --hidden_size=100 --output_size=200 --num_layers=2 --seq_length=21 --pre_gen_model=model_gen --pre_dis_model=dis_pre_model --gen_model=model_1 --dis_model=model_2 --total_epoch=300 --gen_epoch=5 --dis_epoch=3 --latent_size=60 --gpu=2 --dropout=0.5


During adversarisal training, it is very important that we choose a suitable pretrained dismodel.
In the addition, we can change the parameters gen_epoch and dis_epoch.

Note: the taobao data is Chinese, we use the utf-8 code, if you want to use it to deal with the
English, you should change the utf-8 code.












