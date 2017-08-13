
use_valid=0;
%makebatches_mnist;
makebatches;
numbatches_valid=size(testbatchdata,3);
correct = zeros(1,numbatches_valid);
beta = beta0 * soft_plus(WH * M_hidbiasesMax ); 
for tt= 1:numbatches_valid
    data_test = testbatchdata(:,:,tt);
    targets_test = testbatchtargets(:,:,tt);
    p_zy_v_test = P_yz_v(data_test,M_numhid,numclasses,M_hid_visMax,M_hid_yMax,M_hidbiasesMax,M_ybiases,beta,beta0 );
    PP_y_v_test = sum( p_zy_v_test ); %%% 1*C*M  %%%
    P_y_v_test = squeeze(PP_y_v_test);  %%% C*M
    [P_max,target]=max(P_y_v_test);
    [one,TstLbls]=max(targets_test');
    if use_gpu
        TstLbls=gpuArray(TstLbls);
    end
    correct(tt)=sum ( gather(target==TstLbls) );
    
    
end
TestAccuracy=sum(correct)/ncases_test;
fprintf(1, 'Classification error on test set: %6.4f  \n', 1-TestAccuracy);