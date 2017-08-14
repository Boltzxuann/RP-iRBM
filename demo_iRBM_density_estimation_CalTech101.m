 
% Programmed by Xuan Peng
% A illustration of training iRBM with on MNIST
% /Using RP to speed up learning and achieve better generalization
%  
% 2016-2017


initiate

load caltech101_silhouettes_28_split1
maxepoch= 100; %%% 
numclasses= 10;%%
Maxnumhid= 100;%%Initial capacity of oRBM
learning_rate = 1;%%%Ignore it!
use_valid = 1; %%% Use validation set for training 
batchsize = 100;
testbatchsize = 100;
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

%%%%%%Hyper parameters%%%%%%%%%%%
beta0 = 1.01;WH = 0/beta0;  
epW      = 1;   % Learning rate for weights (old ,now useless, same below)
epvb     = 1;   % Learning rate for biases of visible units 
ephb     = 1;   % Learning rate for biases of hidden units 
    
regularization = 'L1'; %%Which regularization is chosen
WC  = 0.001;
use_RP = 1;
h = 1e-10;
p=1;
start_lr = 0.02;
CD= 25;  
PCD = 0;
global_lr = 0.02;   
lr_normal = 0; %%
lr_adaptive=1; adagrad = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

restart=1;
global use_gpu
use_gpu = gpuDeviceCount; 
iRBM; %%%Main code of learning an iRBM.

%%%AIS











