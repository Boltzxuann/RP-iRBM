
if use_valid == 1
      digitdata=Bdata_train(1:50000,:);
      targets=train_targets(1:50000,:);
else
    digitdata=Bdata_train;
    targets=train_targets;
      
    
end

totnum=size(digitdata,1);%%%px:total number of test digits
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

%rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);%%%得到一个所有数据的随机排序
%batchsize = 100;%%%batch大小
numbatches= floor( totnum/batchsize );
numdims  =  size(digitdata,2);%%%每个数据的维数

batchdata = zeros(batchsize, numdims, numbatches);%%%把所有数据转换成若干batches,三维矩阵，每一“层”为一个batch的数据。
batchtargets = zeros(batchsize, numclasses, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);%%%把数据按照前面的随机排序分配给第b个batch。
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets;
%%%%%下面是对测试数据进行同样的操作
if use_valid == 1
     digitdata=Bdata_train(50001:60000,:);
     targets=train_targets(50001:60000,:);
else
    digitdata=Bdata_test;
    targets=test_targets;
end

totnum=size(digitdata,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

%rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);
%testbatchsize = 100;
numbatches= floor( totnum/testbatchsize);
numdims  =  size(digitdata,2);

testbatchdata = zeros(testbatchsize, numdims, numbatches);
testbatchtargets = zeros(testbatchsize, numclasses, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = digitdata(randomorder(1+(b-1)*testbatchsize:b*testbatchsize), :);
  testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*testbatchsize:b*testbatchsize), :);
end;
clear digitdata targets;


%%% Reset random seeds 
% rand('state',sum(100*clock)); 
% randn('state',sum(100*clock)); 



