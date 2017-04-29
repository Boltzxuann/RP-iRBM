
% 
% ��д�ˣ�����/ Programmed by Xuan Peng
% ��MNIST��ѵ��Dis-iRBM��ʾ�������û��Ŀ�꺯��/A illustration of training Dis-iRBM with
%                                            /hybrid ojective on MNIST
% ����RPѵ����������ѧϰ����/Using RP to speed up learning and achieve better
%                         /generalization
% 2016-2017

clear 
close all
addpath('...\...\models')
addpath('...\...\core_modules');
addpath('...\...\evaluation');
addpath('...\...\code_AIS');
addpath('...\...\AIS_iRBM');

load BinaryDataMNIST
maxepoch= 300; %%% Total epochs of training. It takes about 100 ~ 150 epochs to get the best result.
stopepochs = 50; %%The epochs to look ahead for early stopping
numclasses= 10;%%�ܹ���10������
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 1; %%% Use validation set for training 
batchsize = 100;
testbatchsize = 100;
restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
Dis_iRBM__hybrid; %%%Main code of learning a Dis-iRBM.
classification_test;%%% Compute classification error on the test set.











