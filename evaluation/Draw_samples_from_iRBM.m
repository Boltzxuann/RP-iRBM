
%%% Draw samples from the model using Gibbs sampling.

load best_vh 
%load fullmnistvh
vh = gather(vh);
hb = gather(hb);
vb = gather(vb);
global use_gpu
use_gpu = 0;
numdims = size(vh,1);
numcases = 100;
negvisprobs = rand(numcases,numdims);%%%initial states of the chains
negdata = 0.5 > negvisprobs;
numsteps = 10000;
%beta0 = 1.01;
beta = beta0 * soft_plus(0 * hb );
tt = 1;

for   tt = tt : numsteps 
     
     P_z_on_v_neg = P_z( negdata , vh' , hb , numh , beta ,beta0, numcases );  %%% P(z|v)
     
     [Neg_numhidmask, Neg_numhid1hot] = Sample_z (P_z_on_v_neg,numcases,numh); %%% z~P(z|v)
     
     neghidprobs = 1./(  1 + exp( bsxfun(@minus, - vh'* negdata' , hb' ) )  ).*Neg_numhidmask(1:numh,:); %%%P(h|v,z)
     
     neghidstates =  neghidprobs > rand(numh ,numcases); %%% h~P(h|v,z)
     neghidstates = real(neghidstates);
     
     negvisprobs =   1./(   1 + exp( bsxfun( @minus,- vh * neghidstates , vb' ) )  ); %%% P(v|h,z)
     
     negdata = negvisprobs > rand(numdims , numcases);%%% v~P(v|h,z)
     negdata = negdata';
     
     if rem(tt, 500) == 0
        dispims(negdata',28,28);
        drawnow
     end
     fprintf(1,'Gibbs step %d \r',tt);
     
end
      
      