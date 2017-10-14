
% 
% Programmed by Xuan Peng
% A illustration of training Dis-iRBM with hybrid ojective on MNIST
% Using RP to speed up learning and achieve better generalization
% 2016-2017

initiate
load caltech101_silhouettes_28_split1
maxepoch= 200; %%% Total epochs of training. It takes about 100 ~ 150 epochs to get the best result.
stopepochs = inf; %%The number of epochs to look ahead for stopping
numclasses= 101;
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
V=1;
use_valid = V; %%% Use validation set for training 
batchsize = 100;
testbatchsize = 50;
ncases_train = 4100;
ncases_val = 2264;
ncases_test = 2307;
Train_targets = zeros(ncases_train,numclasses);
Val_targets = zeros(ncases_val,numclasses);
test_targets = zeros(ncases_test,numclasses);
for nn = 1:ncases_train
    Train_targets(nn,train_labels(nn)) = 1;
end
for nn = 1:ncases_val
    Val_targets(nn,val_labels(nn)) = 1;
end
for nn = 1:ncases_test
    test_targets(nn,test_labels(nn)) = 1;
end

  %%%%%%Hyper parameters%%%%%%%%
  beta0= 1.01;
  WH = 0/beta0;  
  h = 1e-10;%%%Parameter sqrt(e) in ADAGRAD to avoid dividing 0.
  p = 1;
  start_lr = 0.01;
  global_lr = 0.01;
  a = 0.01; %%Propotion of the generative part
  gen_uselabel= 1;% Use labels for the generative part or not
  regularization = 'L1'; %%Which regularization is chosen: 'no','L1' or 'L2'.
  WC  = 0.001;  %%%Weight decay 
  use_RP = 1;  %%% Whether use RP training or not
  discard = 0;%%%Discard useless hids
  epW      = learning_rate;   % Learning rate for weights 
  ephy      =  learning_rate;   
  ephb       = learning_rate;   
  epyb       = learning_rate;   
  epvb       = learning_rate;
  use_mom = 0; %%% Whether using momentum or not
  mom_inc=0.03;
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











