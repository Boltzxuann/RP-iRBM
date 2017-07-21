ridx = randperm(60000);
if use_valid == 1
      digitdata=Bdata_train(ridx(1:50000),:);
      targets=train_targets(ridx(1:50000),:);
else
    digitdata=Bdata_train;
    targets=train_targets;
      
    
end

totnum=size(digitdata,1);%%%px:total number of test digits
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

%rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);%%%
%batchsize = 100;%%%batch
numbatches= floor( totnum/batchsize );
numdims  =  size(digitdata,2);%%%

batchdata = zeros(batchsize, numdims, numbatches);%%%?
batchtargets = zeros(batchsize, numclasses, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);%%%
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets;
%%%%%下面是对测试数据进行同样的操�?
if use_valid == 1
     digitdata=Bdata_train(ridx(50001:60000),:);
     targets=train_targets(ridx(50001:60000),:);
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



