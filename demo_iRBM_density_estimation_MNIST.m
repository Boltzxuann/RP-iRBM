 
% Programmed by Xuan Peng
% A illustration of training iRBM with on MNIST
% /Using RP to speed up learning and achieve better generalization
%  
% 2016-2017


initiate
load BinaryDataMNIST
maxepoch= 301; %%% 
numclasses= 10;%%
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 1; %%% Use validation set for training 
batchsize = 100;
testbatchsize = 100;
ridx = randperm(60000);%%% Shuffle the training examples
ncases_train = 50000;
train_data = Bdata_train(ridx(1:ncases_train),:);
val_data = Bdata_train(ridx(ncases_train+1:end),:);
test_data = Bdata_test;
Train_targets = train_targets(ridx(1:ncases_train),:);
Val_targets = train_targets(ridx(ncases_train+1:end),:);
%%%%%%Hyper parameters%%%%%%%%%%%
beta0 = 1.01;WH = 0/beta0;  
epW      = 1;   % Learning rate for weights (old ,now useless, same below)
epvb     = 1;   % Learning rate for biases of visible units 
ephb     = 1;   % Learning rate for biases of hidden units 
    
regularization = 'L1'; %%Which regularization is chosen
WC  = 0.0001;
use_RP = 1;
h = 1e-4; %%%Parameter sqrt(e) in ADAGRAD to avoid dividing 0.
p=1;
start_lr = 0.05;
CD= 10;  
PCD = 1;
global_lr = 0.05;   
lr_normal = 0; %%
lr_adaptive=1; adagrad = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
iRBM; %%%Main code of learning an iRBM.

%%%AIS











