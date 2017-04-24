

function  P = P_z( v , hv ,hb , J , beta , beta0, numcases , use_label , y, hy  )

%%%����P(z|v)��P(z|v,y)/Compute P(z|v) or P(z|v,y)

global use_gpu
%use_gpu = gpuDeviceCount;
if nargin<8
    
    [ ln_z , phi_vyz , phi_max] = minus_free_energy( v , hv , hb , J ,beta , beta0 , numcases ); %����-F(v)��-F(z|v) / compute -F(v) and -F(z|v)
else
    [ ln_z , phi_vyz , phi_max] = minus_free_energy( v , hv , hb , J ,beta , beta0 , numcases , use_label , y , hy); %����-F(v)��-F(z|v) / compute -F(v) and -F(z|v)
end

if use_gpu
    P= zeros(J+1,numcases,'gpuArray');
else
    P= zeros(J+1,numcases);
end
if beta0>1
    e_phiRemain_phimax = exp( phi_vyz(J,:)-phi_max ).*exp( (1-beta0)*soft_plus(0) )/(1-exp( (1-beta0)*soft_plus(0) ));
else
    e_phiRemain_phimax = exp( phi_vyz(J,:)- phi_max ).* 100;
end
P(1:J,:) = exp( phi_vyz - repmat(ln_z,J,1 ) );
P(J+1,:) = e_phiRemain_phimax .* exp(-ln_z+phi_max); 

end

