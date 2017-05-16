
% 
% Programmed by Xuan Peng
% A illustration of training Dis-iRBM with hybrid ojective on MNIST
% Using RP to speed up learning and achieve better generalization
% 2016-2017

initiate
load BinaryDataMNIST
maxepoch= 300; %%% Total epochs of training. It takes about 100 ~ 150 epochs to get the best result.
stopepochs = 50; %%The epochs to look ahead for stopping
numclasses= 10;
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 1; %%% Use validation set for training 
batchsize = 100;
testbatchsize = 100;
restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
Dis_iRBM__hybrid; %%%Main code of learning a Dis-iRBM.
%classification_test;%%% Compute classification error on the test set.
Test;











