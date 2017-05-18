
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
randomorder=randperm(totnum);%%%�õ�һ���������ݵ��������
%batchsize = 100;%%%batch��С
numbatches= floor( totnum/batchsize );
numdims  =  size(digitdata,2);%%%ÿ�����ݵ�ά��

batchdata = zeros(batchsize, numdims, numbatches);%%%����������ת��������batches,��ά����ÿһ���㡱Ϊһ��batch�����ݡ�
batchtargets = zeros(batchsize, numclasses, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);%%%�����ݰ���ǰ����������������b��batch��
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets;
%%%%%�����ǶԲ������ݽ���ͬ���Ĳ���
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



