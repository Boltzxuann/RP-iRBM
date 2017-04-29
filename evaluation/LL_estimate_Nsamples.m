
clear
close all
load BinaryDataMNIST
load best_iRBM
global use_gpu
use_gpu = gpuDeviceCount;
ll_sample = zeros(1,5);
for ii=1:5
    random_order_simple;    
    iRBM_AIS_estimate
    ll_sample(ii)= loglik_test_est;
end
ll_mean = mean(ll_sample);
ll_var = var(ll_sample);
