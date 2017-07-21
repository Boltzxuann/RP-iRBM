 
% Programmed by Xuan Peng
% A illustration of training iRBM with on MNIST
% /Using RP to speed up learning and achieve better generalization
%  
% 2016-2017


initiate
load BinaryDataMNIST
%load caltech101_silhouettes_28_split1
maxepoch= 100; %%% 
numclasses= 10;%%
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 1; %%% Use validation set for training 
ridx = randperm(60000);%%% Random permutate the training examples
batchsize = 100;
testbatchsize = 100;
restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
iRBM; %%%Main code of learning an iRBM.

%%%AIS











