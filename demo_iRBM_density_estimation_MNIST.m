 
% 编写人：彭轩/ Programmed by Xuan Peng
% 在MNIST上训练Dis-iRBM的示例，采用混合目标函数/A illustration of training Dis-iRBM with
%                                            /hybrid ojective on MNIST
% 采用RP训练方法加速学习过程/Using RP to speed up learning and achieve better
%                         /generalization
% 2016-2017

clear 
close all
addpath('...\...\models')
addpath('...\...\core_modules');
addpath('...\...\evaluation');
addpath('...\...\AIS_RBM');
addpath('...\...\AIS_iRBM');

load BinaryDataMNIST
%load caltech101_silhouettes_28_split1
maxepoch= 100; %%% Total epochs of training. It takes about 100 ~ 150 epochs to get the best result.
numclasses= 10;%%总共有10类数据
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 1; %%% Use validation set for training 
batchsize = 100;
testbatchsize = 100;
restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
iRBM; %%%Main code of learning an iRBM.

%%%AIS的代码











