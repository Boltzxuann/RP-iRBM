%%%Computing P(z,y|v) for a mini-batch of data.
%%%直接求P(z,y|v),尽可能的矩阵（张量）化运算，去掉了所有的循环。


 if use_gpu  == 1
      p_zy_on_v = zeros(J+1, numclasses , numcases,   'gpuArray');
 else
      p_zy_on_v = zeros(J+1, numclasses , numcases );
 end
 %  for k =1 : numcases
       hv_d_c = repmat(hid_visMax(1:J,:)*data',[1 1 numclasses]);
     
       hv_c_M = permute(hv_d_c,[1 3 2]); 
       
    F_sumterm_zy = soft_plus(   bsxfun(  @plus,  hv_c_M, ...
                                bsxfun( @plus, hid_yMax(1:J,:) ,hidbiasesMax(1:J)')  )   )...
                     -repmat(beta(1:J)',[1 numclasses numcases]);  %%% J * C * M
     sum_FsumTst = cumsum( F_sumterm_zy );  %%% J * C * M
     
     phi_zy  = bsxfun( @plus,ybiases, sum_FsumTst); %%% J * C * M
     max_phi_zy = max( max (phi_zy) ); %%%  1 * M
     
     e_phi_maxphi =  exp(   bsxfun(@minus, phi_zy, max_phi_zy )     )  ;
     if beta0>1
         e_phiRemain_phimax_yz = e_phi_maxphi(J,:,:) .* exp( (1-beta0)*soft_plus(0) )/(1-exp( (1-beta0)*soft_plus(0) ));
     else
         e_phiRemain_phimax_yz = e_phi_maxphi(J,:,:) * 100 ;  %%% 1 * C * M   %%%取巧
     end
     if J ==1
          ln_zy=  max_phi_zy + log( sum ( ( e_phi_maxphi ) + e_phiRemain_phimax_yz )  ); %%%  1 * 1 * M

     else
          ln_zy = max_phi_zy + log(  sum( sum( e_phi_maxphi ) + e_phiRemain_phimax_yz  ) ); %%% 1 * 1 * M

     end
     
   %  p_zy_on_v(1:J,:,:) = bsxfun( @times, e_phi_maxphi,  exp(repmat(-ln_zy,1,numclasses )+ max_phi_zy) );
     p_zy_on_v(1:J,:,:) = exp( bsxfun( @plus, phi_zy, -ln_zy) );
     if beta0>1
         %p_zy_on_v(J+1,:,:) =  e_phiRemain_phimax_yz .* exp(-ln_zy+max_phi_zy);
         p_zy_on_v(J+1,:,:) =  bsxfun(@times, e_phiRemain_phimax_yz , exp(-ln_zy+max_phi_zy) );
     else
         p_zy_on_v(J+1,:,:) =  p_zy_on_v(J,:,:) *100; %%%取巧
     end
     
%   end
