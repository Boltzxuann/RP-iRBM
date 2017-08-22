
% 
% Programmed by Xuan Peng
% A illustration of training Dis-iRBM with hybrid ojective on MNIST
% Using RP to speed up learning and achieve better generalization
% 2016-2017

initiate
load BinaryDataMNIST
maxepoch= 200; %%% Total epochs of training. It takes about 100 ~ 150 epochs to get the best result.
stopepochs = inf; %%The number of epochs to look ahead for stopping
numclasses= 10;
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 0; %%% Use validation set for training or not
batchsize = 200;
testbatchsize = 100;
ridx = randperm(60000);%%% Shuffle the training examples
ncases_train = 60000;
ncases_val = 60000-ncases_train;
ncases_test = 10000;
train_data = Bdata_train(ridx(1:ncases_train),:);
val_data = Bdata_train(ridx(ncases_train+1:end),:);
test_data = Bdata_test;
Train_targets = train_targets(ridx(1:ncases_train),:);
Val_targets = train_targets(ridx(ncases_train+1:end),:);

  %%%%%%Hyper parameters%%%%%%%%
  beta0= 1.01;
  WH = 0/beta0;  
  h = 1e-10;%%%Parameter sqrt(e) in ADAGRAD to avoid dividing 0.
  p = 1;
  start_lr = 0.1;
  global_lr = 0.1;
  a = 0.01; %%Propotion of the generative part
  gen_uselabel= 1;% Use labels for the generative part or not
  regularization = 'L1'; %%Which regularization is chosen: 'no','L1' or 'L2'.
  WC  = 0.00005;  %%%Weight decay 
  use_RP = 1;  %%% Whether use RP training or not
  discard = 0;%%%Discard useless hids
  epW      = learning_rate;   % Learning rate for weights 
  ephy      =  learning_rate;   
  ephb       = learning_rate;   
  epyb       = learning_rate;   
  epvb       = learning_rate;
  use_mom = 1; %%% Whether using momentum or not
  G =0;
  CD = 3;
  label = 1;
  lr_normal = 0; 
  lr_adaptive=1; adagrad = 1;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
Dis_iRBM__hybrid; %%%Main code of learning a Dis-iRBM.
%classification_test;%%% Compute classification error on the test set.
Test;











