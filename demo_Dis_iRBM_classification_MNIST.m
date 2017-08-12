
% 
% Programmed by Xuan Peng
% A illustration of training Dis-iRBM with hybrid ojective on MNIST
% Using RP to speed up learning and achieve better generalization
% 2016-2017

initiate
load BinaryDataMNIST
maxepoch= 300; %%% Total epochs of training. It takes about 100 ~ 150 epochs to get the best result.
stopepochs = 50; %%The number of epochs to look ahead for stopping
numclasses= 10;
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 1; %%% Use validation set for training 
batchsize = 200;
testbatchsize = 100;
ridx = randperm(60000);%%% Shuffle the training examples
ncases_train = 50000;
train_data = Bdata_train(ridx(1:ncases_train),:);
val_data = Bdata_train(ridx(ncases_train+1:end),:);
Train_targets = train_targets(ridx(1:ncases_train),:);
Val_targets = train_targets(ridx(ncases_train+1:end),:);

restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
Dis_iRBM__hybrid; %%%Main code of learning a Dis-iRBM.
%classification_test;%%% Compute classification error on the test set.
Test;











