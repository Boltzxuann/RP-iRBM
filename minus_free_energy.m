function  [ minus_F_v  , minus_F_z , phi_max ]= minus_free_energy ( v,hv,hb,J,beta,beta0, numcases , use_label, y , hy)

%计算-F(v)及-F(z|v) / compute -F(v) and -F(z|v)

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
                                  
  minus_F_z =  cumsum(F_sumterm); %%%计算F（z,v,y）中的求和项中的每个元素,一次把一个batch的都计算了,每一列对应batch中的一个样本。这是一个J*D的矩阵。
  
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