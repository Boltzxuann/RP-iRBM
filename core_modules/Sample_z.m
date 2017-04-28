

function [ numhid_mask, numhid_1hot] = Sample_z( Pz , numcases, J)

%%% 从P(z|v)或P(z|v，y)中采样得到z/Sample z from P(z|v) or P(z|v，y)

global use_gpu
%use_gpu = gpuDeviceCount;
if use_gpu
    rand_01 = repmat(rand(1,numcases,'gpuArray'),J+1,1);
else
    rand_01 = repmat(rand(1,numcases),J+1,1);
end
cumsum_Pz = cumsum(Pz);
Pos_numhid1 = cumsum_Pz > rand_01;
Pos_numhid2 = cumsum_Pz < rand_01;
if use_gpu
    numhid_mask = zeros(J+1,numcases,'gpuArray');
else
    numhid_mask = zeros(J+1,numcases);
end
numhid_mask(1,:) = 1; 
numhid_mask(2:J+1,:) = Pos_numhid2(1:J,:);
numhid_1hot = Pos_numhid1.* numhid_mask;

end