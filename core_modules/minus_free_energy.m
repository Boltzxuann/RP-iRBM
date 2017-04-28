function  [ minus_F_v  , minus_F_z , phi_max ]= minus_free_energy ( v,hv,hb,J,beta,beta0, numcases , use_label, y , hy)

%����-F(v)��-F(z|v) / compute -F(v) and -F(z|v)

  if nargin<8
      use_label = 0;
  end
  if use_label 
      F_sumterm = soft_plus( bsxfun( @plus, (hv(1:J,:)* v' + hy(1:J,:)*y') , hb(1:J)')  )...
                  - repmat(beta(1:J)',1,numcases);   %%% J*numcases 
  else
      F_sumterm = soft_plus(  bsxfun( @plus, hv(1:J,:)* v' , hb(1:J)')  )...
                  - repmat(beta(1:J)',1,numcases);   %%% J*numcases   
  end
                                  
  minus_F_z =  cumsum(F_sumterm); %%%����F��z,v,y���е�������е�ÿ��Ԫ��,һ�ΰ�һ��batch�Ķ�������,ÿһ�ж�Ӧbatch�е�һ������������һ��J*D�ľ���
  
  if J ==1 
      phi_max = minus_F_z(1,:);
  else
      phi_max = max(minus_F_z,[],1);
  end
  phi_phimax = minus_F_z - repmat(phi_max, J, 1);
  if beta0 > 1
      e_phiRemain_phimax = exp( minus_F_z(J,:)-phi_max ).*exp( (1-beta0)*soft_plus(0) )/(1-exp( (1-beta0)*soft_plus(0) ));
  else
      e_phiRemain_phimax = exp( minus_F_z(J,:)- phi_max ).* 100;
  end
  if J ==1
      minus_F_v =  phi_max + log( ( exp( phi_phimax) ) + e_phiRemain_phimax );
  else
      minus_F_v = phi_max + log( sum( exp( phi_phimax) ) + e_phiRemain_phimax );
  end
  
end