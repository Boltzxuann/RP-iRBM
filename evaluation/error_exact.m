
numbatches_valid=size(batchdata,3);
correct = zeros(1,numbatches_valid);
for tt= 1:numbatches_valid
    data_val = batchdata(:,:,tt);
    targets_val = batchtargets(:,:,tt);
    p_zy_v_valid = P_yz_v(data_val,numh,numclasses,hid_visMax,hid_yMax,hidbiasesMax,ybiases,beta,beta0 );
    PP_y_v_val = sum( p_zy_v_valid ); %%% 1*C*M  %%%
    P_y_v_val = squeeze(PP_y_v_val);  %%% C*M
    [P_max,target]=max(P_y_v_val);
    [one,TstLbls]=max(targets_val');
    if use_gpu
        TstLbls=gpuArray(TstLbls);
    end
    correct(tt)=sum ( gather(target==TstLbls) );
    
    
end
TestAccuracy=sum(correct)/ncases_train;