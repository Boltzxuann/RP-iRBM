%Training the Dis-iRBM with Hybrid objective
%By Xuan Peng
% 2016-2017
%
%




  
%%%% Initiate the parameters %%%%  
if restart ==1
  restart=0;
  epoch=1;
  M_epoch=1;
  %makebatches_mnist;
  makebatches;
  [numcases, numdims, numbatches]=size(batchdata);
  if use_gpu
      batchdata=gpuArray(batchdata);
      batchtargets=gpuArray(batchtargets);
  end
  
  max_ValAccy = 0;
  test_epoch = zeros( 2,maxepoch );
  if use_mom
      initialmomentum  = 0.0;
  else
      initialmomentum  = 0.5;
  end
  mom = initialmomentum * ones( Maxnumhid,1 );
  momentum=initialmomentum;
  lr = 1*ones( Maxnumhid,1 );
  num_ini = 10;%%% Use normal lr for first n steps. 
  initial = ones(1,Maxnumhid) * num_ini;
  start = 1; %%% The learning of parameters not related to hidden units can be slower.
  
  if use_gpu
      weightcost= zeros(Maxnumhid,numdims,'gpuArray');
      hycost=zeros(Maxnumhid,numclasses, 'gpuArray'); 
  
      epsilonW = zeros(Maxnumhid,numdims, 'gpuArray');
      epsilon_hy = zeros(Maxnumhid,numclasses, 'gpuArray'); 
      epsilonhb  = zeros(1,Maxnumhid, 'gpuArray');
      epsilonyb  =  zeros(1,numclasses, 'gpuArray');       
      epsilonvb  = zeros(1,numdims, 'gpuArray');
  
      LzW = zeros(Maxnumhid,numdims, 'gpuArray');
      LzU = zeros(Maxnumhid,numclasses, 'gpuArray'); 
      LzHb  = zeros(1,Maxnumhid, 'gpuArray');
  else
      weightcost= zeros(Maxnumhid,numdims);
      hycost=zeros(Maxnumhid,numclasses); 
  
      epsilonW = zeros(Maxnumhid,numdims);
      epsilon_hy = zeros(Maxnumhid,numclasses); 
      epsilonhb  = zeros(1,Maxnumhid);
      epsilonyb  =  zeros(1,numclasses);       
      epsilonvb  = zeros(1,numdims);
  
      LzW = zeros(Maxnumhid,numdims);
      LzU = zeros(Maxnumhid,numclasses); 
      LzHb  = zeros(1,Maxnumhid);
      
  end
  J             = 2;%%%
  J_r = 1;
  Max_J_r = 1;
  mean_maxPN_epoch = J; 
  if use_gpu
      visbiases     = zeros(1,numdims, 'gpuArray');%%%%
      ybiases      = zeros(1,numclasses, 'gpuArray');            %%% 
      hid_visMax    = 0.0*randn(Maxnumhid,numdims, 'gpuArray');      
      hid_yMax      = 0.0*randn(Maxnumhid,numclasses, 'gpuArray');   
      hidbiasesMax  = 0* ones(1,Maxnumhid, 'gpuArray');           %%%
      
      hid_visMax_inc  = zeros(Maxnumhid,numdims, 'gpuArray');
      hid_yMax_inc    = zeros(Maxnumhid,numclasses, 'gpuArray');
  
      hidbiasesMaxinc   = zeros(1,Maxnumhid, 'gpuArray');
      ybiases_inc    = zeros(1,numclasses, 'gpuArray');
      visbiases_inc  = zeros(1,numdims, 'gpuArray');
      
      poshidprobs = zeros(numcases,Maxnumhid, 'gpuArray');%%%
      neghidprobs = zeros(numcases,Maxnumhid, 'gpuArray');%%%
      posprods    = zeros(numdims,Maxnumhid, 'gpuArray');
      negprods    = zeros(numdims,Maxnumhid, 'gpuArray');
      
      grad_W_history = zeros(Maxnumhid , numdims, 'gpuArray');
      grad_U_history = zeros(Maxnumhid, numclasses, 'gpuArray');
      grad_hb_history = zeros(1,Maxnumhid, 'gpuArray');
      grad_yb_history = zeros(1, numclasses, 'gpuArray');
      grad_vb_history = zeros(1,numdims, 'gpuArray');
 
      W_in_history = zeros(Maxnumhid , numdims, 'gpuArray');
      hb_in_history = zeros(1,Maxnumhid, 'gpuArray');
      yb_in_history = zeros(1, numclasses, 'gpuArray');
      U_in_history = zeros(Maxnumhid, numclasses, 'gpuArray');
      vb_in_history = zeros(1, numdims, 'gpuArray');
  else
      
      visbiases     = zeros(1,numdims);%%%%
      ybiases      = zeros(1,numclasses);            %%%
      hid_visMax    = zeros(Maxnumhid,numdims);       %%%
      hid_yMax      = zeros(Maxnumhid,numclasses);    %%%
      hidbiasesMax  = zeros(1,Maxnumhid);           %%%
      
      hid_visMax_inc  = zeros(Maxnumhid,numdims);
      hid_yMax_inc    = zeros(Maxnumhid,numclasses);
  
      hidbiasesMaxinc   = zeros(1,Maxnumhid);
      ybiases_inc    = zeros(1,numclasses);
      visbiases_inc  = zeros(1,numdims);
      
      poshidprobs = zeros(numcases,Maxnumhid);%%%
      neghidprobs = zeros(numcases,Maxnumhid);%%%
      posprods    = zeros(numdims,Maxnumhid);
      negprods    = zeros(numdims,Maxnumhid);
      
      grad_W_history = zeros(Maxnumhid , numdims);
      grad_U_history = zeros(Maxnumhid, numclasses);
      grad_hb_history = zeros(1,Maxnumhid);
      grad_yb_history = zeros(1, numclasses);
      grad_vb_history = zeros(1,numdims);

      W_in_history = zeros(Maxnumhid , numdims);
      hb_in_history = zeros(1,Maxnumhid);
      yb_in_history = zeros(1, numclasses);
      U_in_history = zeros(Maxnumhid, numclasses);
      vb_in_history = zeros(1, numdims);
  
  end
 
   %targets_batch=batchtargets(:,:,1);%%%
   if use_gpu
      negtargets_gen = zeros(numcases,numclasses,'gpuArray');
      targets_batch_dis= zeros(numcases,numclasses,'gpuArray');
      neg_numhid_dis = ones(1,numcases,'gpuArray');
      neg_numhid_gen = ones(1,numcases,'gpuArray');
      negDstates = rand(numcases,numdims,'gpuArray');%%%start states of the chains
      negdata = 0.3 > negDstates;
      negdata = gpuArray(round(negdata));
   else
      negtargets_gen = zeros(numcases,numclasses);
      targets_batch_dis= zeros(numcases,numclasses);
      neg_numhid_dis = ones(1,numcases);
      neg_numhid_gen = ones(1,numcases);
      negDstates = rand(numcases,numdims);%%%start states of the chains
      negdata = 0.3 > negDstates;
   end
   
end

for epoch = epoch:maxepoch
    
    if lr_normal
       learning_rate = 1/ceil(epoch/20); %%% learning rate decay
       %learning_rate = 0.5;
       learning_rate = max( learning_rate , 0.01);
       epW       = learning_rate;   % Learning rate for weights 
       ephy      =  learning_rate;   

       ephb       = learning_rate;  
       epyb       = learning_rate;   
       epvb       = learning_rate;
     
    end
        
%      lr(1:mean_maxPN_epoch) = lr(1:mean_maxPN_epoch)*0.99;
%      lr = max(lr,0.01);

    %makebatches_mnist;
    makebatches;
    [numcases, numdims, numbatches]=size(batchdata);
    if use_gpu
        batchdata=gpuArray(batchdata);
        batchtargets = gpuArray(batchtargets);
    end

    fprintf(1,'epoch %d\r',epoch); 
    errsum=0;
     Maxnegnumhid = zeros(1,numbatches);
     Maxposnumhid = zeros(1,numbatches);
     Meannegnumhid = zeros(1,numbatches);
     Meanposnumhid = zeros(1,numbatches);
     Minposnumhid = zeros(1,numbatches);
     
   for batch = 1:numbatches
      if J >= Maxnumhid -1   %%%% 
           Maxnumhid = Maxnumhid + 100;
        
           hid_visMax(J+1:Maxnumhid,:)    = 0.0; 
           hid_yMax (J+1:Maxnumhid,:)      = 0.0;
           hidbiasesMax(J+1:Maxnumhid)   = -0;  
        
           hid_visMax_inc(J+1:Maxnumhid,:)  = 0;
           hid_yMax_inc(J+1:Maxnumhid,:)    = 0;
           hidbiasesMaxinc(J+1:Maxnumhid)   = 0;
           
           epsilonW(J+1:Maxnumhid,:) = 0;
           epsilon_hy(J+1:Maxnumhid,:) = 0; 
           epsilonhb(J+1:Maxnumhid)  = 0;
        
           grad_W_history(J+1:Maxnumhid,:) = 0;
           grad_U_history(J+1:Maxnumhid,:) = 0;
           grad_hb_history(J+1:Maxnumhid)= 0;
                           
           W_in_history(J+1:Maxnumhid,:) =  0;
           U_in_history(J+1:Maxnumhid,:)=  0;
           hb_in_history(J+1:Maxnumhid)=  0;
           if use_gpu
               weightcost= zeros(Maxnumhid,numdims,'gpuArray');
               hycost=zeros(Maxnumhid,numclasses,'gpuArray'); 
           else
               weightcost= zeros(Maxnumhid,numdims);
               hycost=zeros(Maxnumhid,numclasses); 
           end
           
           mom(J+1:Maxnumhid)= initialmomentum;
           initial(J+1:Maxnumhid) =  num_ini;
      end  
    

      if use_RP
          random_order_simple;
      end
   
      lwc = linspace(1,1, J )';
      lr = linspace(1,1, Maxnumhid )';
      bt = linspace(1,1, Maxnumhid );
      
      epsilonW             = epW .* repmat( lr ,1, numdims );   % Learning rate for weights 
      epsilon_hy           = ephy.* repmat( lr ,1,numclasses );   %
      epsilonhb            = ephb.* lr';   %
      epsilonyb            = epyb;   %
      epsilonvb            = epvb;
      weightcost(1:J,:)    = WC* repmat( lwc ,1, numdims );
      hycost(1:J,:)        = WC* repmat( lwc , 1, numclasses );

      if epoch ==1
          eff_nh = J;
      else
          eff_nh = numh;%length(hbMax);
      end
      fprintf(1,'epoch %d batch %d z %d effective hids %d \r',epoch, batch, J, eff_nh); 

 
%%%%%%%%% POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      data = batchdata(:,:,batch);%%%
      targets_0 = batchtargets(:,:,batch);   
      
      beta = beta0 * bt .* soft_plus(WH * hidbiasesMax ); 
      [~,y_0] = max(targets_0,[],2);   
      P_z_on_vy = P_z( data , hid_visMax ,hidbiasesMax , J , beta ,beta0 ,numcases , 1 , targets_0, hid_yMax ); %%%compute P(z|v,y)   
      [Pos_numhid_mask, Pos_numhid_1hot ] = Sample_z (P_z_on_vy,numcases,J); %%% sample z from P(z|v,y)
      
      [~,Pos_numhid] = max(Pos_numhid_1hot);
      where_Jpp = find(Pos_numhid==J+1);
      Pos_numhid_1 = Pos_numhid;
      Pos_numhid_1(where_Jpp)=1;
      M_pnh=max(Pos_numhid_1);
      %M_pnh=max(Pos_numhid);

      sum_P_z_on_vy = cumsum(P_z_on_vy);
      if use_gpu
          Pos_MaxNh =zeros(Maxnumhid,numcases,'gpuArray');
          S_PzOnvy = ones(Maxnumhid,numcases,'gpuArray');
      else
          Pos_MaxNh =zeros(Maxnumhid,numcases);
          S_PzOnvy = ones(Maxnumhid,numcases);
      end
      S_PzOnvy(1:J,:) = sum_P_z_on_vy(1:J,:);
      
      Pos_MaxNh(1:J,:)= 1;

%      Pos_numhid4 =zeros(Maxnumhid,numcases);
%      Pos_numhid4(1:J+1,:)= Pos_numhid3;
      %poshidprobs = 1./(  1 + exp(  - pagefun(@mtimes,hid_visMax  ,data')  -  pagefun(@mtimes,hid_yMax,targets_0')  - repmat( hidbiasesMax',1,numcases ) )  );
      poshidprobs = 1./(  1 + exp(  - hid_visMax * data' -  hid_yMax * targets_0' - repmat( hidbiasesMax',1,numcases ) )  );
      % poshidprobs = poshidprobs.*Pos_numhid4;
      %poshidprobs = poshidprobs.* Pos_MaxNh;
      poshidprobs = poshidprobs.* (1-S_PzOnvy);%%%Directly compute p(h|v,y)

%      Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )) .*Pos_numhid4;
      %Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )) .*Pos_MaxNh;
      Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )) .* (1-S_PzOnvy);
      poshidprobsMinusDbeta = ( poshidprobs-  Dbeta_h);
 
      % poshidprobsMinusDbeta = poshidprobsMinusDbeta.* Pos_numhid4;

      %posprods    = poshidprobs * data;%%%v(0)t1*p(h|v(0)t1)+v(0)t2*p(h|v(0)t2)+...+v(0)tm*p(h|v(0)tM) 
      posprods_dis    = poshidprobs * data  ;
      %posprods_dis   = poshidprobs .*( 1-  S_PzOnvy ) * data   ;
      %batchposhidprobs(:,:,batch)=poshidprobs;
      pos_hidy_dis    = poshidprobs * targets_0;%%%
      %pos_hidy_dis   = poshidprobs .*( 1-  S_PzOnvy ) * targets_0 ;
      poshidact_dis   = sum(poshidprobsMinusDbeta , 2).';%%%
      %poshidact_dis   = sum( poshidprobsMinusDbeta .*( 1-  S_PzOnvy ) , 2 ).';
      posvisact_dis   = sum( data );%%%
      %posvisact_2 = sum(data.^2);%%%
      posyact_dis     = sum(targets_0);%%%
      
      if gen_uselabel
          posprods_gen = posprods_dis;
          poshidact_gen = poshidact_dis;
          posvisact_gen = posvisact_dis;
          pos_hidy_gen = pos_hidy_dis;
          posyact_gen = posyact_dis;
      else
          
          P_z_on_v = P_z( data , hid_visMax ,hidbiasesMax , J , beta ,beta0 ,numcases ); %% æ²¡æç¨æ ç­?/ without labels for the generative part
          sum_P_z_on_v = cumsum(P_z_on_v);
          [~, Pos_numhid1hot] = Sample_z (P_z_on_v,numcases,J);
          [~,Pos_numhid_gen] = max(Pos_numhid1hot);
          M_pnh_gen=max(Pos_numhid_gen);
         
          if use_gpu
              Pos_MaxNh =zeros(Maxnumhid,numcases,'gpuArray');
              S_PzOnv = ones(Maxnumhid,numcases,'gpuArray');
          else
              Pos_MaxNh =zeros(Maxnumhid,numcases);         
              S_PzOnv = ones(Maxnumhid,numcases);
          end
          Pos_MaxNh(1:J,:)= 1;
          S_PzOnv(1:J,:) = sum_P_z_on_v(1:J,:);
  
          %poshidprobs_gen = 1./(1 + exp(-pagefun(@mtimes,hid_visMax  ,data') - repmat( hidbiasesMax',1,numcases )));
          poshidprobs_gen = 1./(1 + exp(-hid_visMax * data' - repmat( hidbiasesMax',1,numcases )));
         % poshidprobs = poshidprobs.*Pos_numhid4;
          %poshidprobs_gen = poshidprobs_gen.*Pos_MaxNh;
          poshidprobs_gen = poshidprobs_gen.* (1-S_PzOnv);%%%Directly compute p(h|v)
          %  Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )) .*Pos_numhid4;
          %Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )) .*Pos_MaxNh;
          Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )) .* (1-S_PzOnv);
          poshidprobsMinusDbeta_gen = (poshidprobs_gen -  Dbeta_h);
 
          % poshidprobsMinusDbeta = poshidprobsMinusDbeta.* Pos_numhid4;

%           posprods_gen   = poshidprobs_gen .*( 1-  S_PzOnv ) * data   ;
%           poshidact_gen   = sum( poshidprobsMinusDbeta_gen .*( 1-  S_PzOnv ) , 2 ).';
          posprods_gen   = poshidprobs_gen * data   ;
          poshidact_gen   = sum( poshidprobsMinusDbeta_gen , 2 ).';
          posvisact_gen   = sum(data);%%%
          
      end

%%%%%%%% negtive phase of generative part%%%%%%%%%%
     if 1 %%CD or PCD
        negtargets_gen = batchtargets(:,:,batch);
        negdata = data;
     end
     for cditer=1:CD
     %%%%%%% z~p(z|v)%%%%%%%%%%%%%

         P_z_on_vy_neg = P_z( negdata , hid_visMax ,hidbiasesMax , J , beta ,beta0, numcases , gen_uselabel , negtargets_gen, hid_yMax ); 
         sum_P_z_on_vy_neg = cumsum(P_z_on_vy_neg);
         [Neg_numhidmask, Neg_numhid1hot] = Sample_z (P_z_on_vy_neg,numcases,J);
         [~,neg_numhid_gen] = max(Neg_numhid1hot); 
         M_nh_gen = max(neg_numhid_gen);
%          neg_MaxNh =zeros(Maxnumhid,numcases);
%          neg_MaxNh(1: J,:)= 1;
%          if use_gpu
%              s_negPzONvy = ones(Maxnumhid,numcases,'gpuArray');
%          else
%              s_negPzONvy = ones(Maxnumhid,numcases);
%          end
%          s_negPzONvy(1:J,:) = sum_P_z_on_vy_neg(1:J,:);
   
      %%%%%%% z~p(z|v)%%%%%%%%%%%%%
      
      
     %%%%%%% h~p(h|v,z)%%%%%%%%%%%%% 
         if use_gpu    
             neg_numhid4 =zeros(Maxnumhid,numcases,'gpuArray');
         else
             neg_numhid4 =zeros(Maxnumhid,numcases);
         end
         neg_numhid4(1:J+1,:)= Neg_numhidmask;
         if gen_uselabel
             %neghidprobs = 1./(1 + exp(   pagefun(@mtimes, -hid_visMax, negdata') +  pagefun(@mtimes ,- hid_yMax , negtargets_gen .') - repmat( hidbiasesMax',1,numcases )));  
             neghidprobs = 1./(1 + exp(   -hid_visMax * negdata'   - hid_yMax * negtargets_gen .' - repmat( hidbiasesMax',1,numcases )));  
         else   
             %neghidprobs = 1./(1 + exp(   pagefun(@mtimes, -hid_visMax, negdata') - repmat( hidbiasesMax',1,numcases )));  
             neghidprobs = 1./(1 + exp(  -hid_visMax * negdata'  - repmat( hidbiasesMax',1,numcases )));  
         end
       if use_gpu
           S_Pz_neg = ones(Maxnumhid,numcases,'gpuArray');
       else
           S_Pz_neg = ones(Maxnumhid,numcases);
       end
       S_Pz_neg(1:J,:) =  sum_P_z_on_vy_neg(1:J,:);
       %neghidprobs = neghidprobs.* (1-S_Pz_neg); %%% Compute p(v|h) directly.
       neghidprobs = neghidprobs.* neg_numhid4; %%%
  
       neghidstates =  round( neghidprobs > rand(Maxnumhid ,numcases) );
       %neghidstates = neghidstates.* neg_numhid4;%%% z is still needed.
  
    %%%%%% v~p(v|h,z)%%%%%%%%%%%%% 
   
        % negvisprobs =   1./(1 + exp( - pagefun(@mtimes, hid_visMax' ,  neghidstates ) - repmat( visbiases',1,numcases )));      
         negvisprobs =   1./(1 + exp(- hid_visMax'* neghidstates  - repmat( visbiases',1,numcases )));      
         negdata = negvisprobs > rand(numdims , numcases);
         negdata = real ( negdata.');
  
         if gen_uselabel
             phi_y = neghidstates'*hid_yMax +repmat(ybiases,numcases,1);
             %phi_y = bsxfun(  @plus, pagefun(@mtimes,  neghidstates' , hid_yMax ), ybiases  );
             phi_y_max = max(phi_y,[],2);
             phiy_phiymax = phi_y - repmat(phi_y_max,1,numclasses);
             ln_zy = phi_y_max + log( sum(exp(phiy_phiymax),2) );

             negtargetsprobs = exp(phi_y-repmat(ln_zy,1,numclasses));
             % phi_tgt_on_h = exp(neghidstates'*hid_yMax +repmat(ybiases,numcases,1));
             %sum_phi_tonh = sum(phi_tgt_on_h,2);
             %negtargetsprobs = phi_tgt_on_h./repmat(sum_phi_tonh,1,numclasses);
             rand_0_1=repmat(rand(numcases,1),1,numclasses );
             sum_negtgtprobs=(cumsum(negtargetsprobs'))';
             negtargets_1 = sum_negtgtprobs > rand_0_1;
             negtargets_2 = sum_negtgtprobs < rand_0_1;
             if use_gpu
                 negtargets_3 = zeros(numcases,numclasses,'gpuArray');
             else
                 negtargets_3 = zeros(numcases,numclasses);
             end
             negtargets_3(:,1)=1;
             negtargets_3(:,2:numclasses)=negtargets_2(:,1:numclasses-1);
             negtargets_gen = negtargets_1.*negtargets_3;
         end
         
 %%%%%%%%%%%%% Reconstructing error %%%%%%%%%%%
        if cditer == 1
          if gen_uselabel
             err= sum(sum( abs(targets_0 - negtargets_gen)/2 ));%%%éæè¯¯å·®
          else
             err= sum(sum( abs(data - negdata) ));%%%éæè¯¯å·®
          end
          errsum = err + errsum;   
        end
         
     end
    if use_gpu
        neg_MaxNh =zeros(Maxnumhid,numcases,'gpuArray');
        s_negPzONvy = ones(Maxnumhid,numcases,'gpuArray');
    else
        neg_MaxNh =zeros(Maxnumhid,numcases);
        s_negPzONvy = ones(Maxnumhid,numcases);
    end
    P_z_on_vy_neg = P_z( negdata , hid_visMax ,hidbiasesMax , J , beta ,beta0, numcases , gen_uselabel , negtargets_gen, hid_yMax ); 
    sum_P_z_on_v_neg = cumsum(P_z_on_vy_neg);
    neg_MaxNh(1:J,:)= 1;
    s_negPzONvy(1:J,:) = sum_P_z_on_v_neg(1:J,:);     
     
     if gen_uselabel %%% if ever use labels for the generative part 
         neghidprobs = 1./(1 + exp(- hid_visMax* negdata' - hid_yMax* negtargets_gen .' - repmat(hidbiasesMax',1,numcases )));   
     else
         neghidprobs = 1./(1 + exp(- hid_visMax* negdata'  - repmat(hidbiasesMax',1,numcases )));    
     end
     %   neghidprobs = 1./(1 + exp(   pagefun(@mtimes, -hid_visMax, negdata') +  pagefun(@mtimes ,- hid_yMax , negtargets_gen .') - repmat( hidbiasesMax',1,numcases ))); 
        %neghidprobs = 1./(1 + exp(   pagefun(@mtimes, -hid_visMax, negdata') - repmat( hidbiasesMax',1,numcases )));  
     %neghidprobs = neghidprobs.* neg_MaxNh;
     neghidprobs = neghidprobs.* (1-s_negPzONvy);
     %Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )).* neg_MaxNh;
     Dbeta_h =    1 * WH * beta0 * 1./(1+ exp(-WH * repmat( hidbiasesMax',1,numcases ) )).* (1-s_negPzONvy);
     neghidprobsMinusDbeta = neghidprobs-  Dbeta_h;
    
%      negprods_gen    = neghidprobs .*( 1- s_negPzONvy ) * negdata ;
%      neghidact_gen   = sum( neghidprobsMinusDbeta.*( 1- s_negPzONvy )  ,2 ).'; 
     negprods_gen    = neghidprobs * negdata ;
     neghidact_gen   = sum( neghidprobsMinusDbeta  ,2 ).';  
     negvisact_gen = sum(negdata); 
       
    if gen_uselabel
       negyact_gen = sum(negtargets_gen);
       %neg_hidy_gen    = neghidprobs.*( 1- s_negPzONvy ) * negtargets_gen;
       neg_hidy_gen    = neghidprobs * negtargets_gen;
    end
  
%      negyact = sum(negtargets);
%      neg_hidy = neghidprobs * negtargets;

%%%%% Compute negtive phase of disciminative part %%%%%%
       compute_P_yz_v;
       PP_y_v = sum( p_zy_on_v ); %%% 1*C*M  %%%
       P_y_v = squeeze(PP_y_v);  %%% C*M
  

       hv_c = repmat(hid_visMax, 1,1,numclasses );  %%%  J * V *C
       if use_gpu
           hd_c  = pagefun(@mtimes, hv_c, data');  %%%
       else
           hd_c = zeros(Maxnumhid,numcases,numclasses);
           for cc = 1:numclasses
               hd_c(:,:,cc)= hv_c(:,:,cc)*data';
           end
       end
       h_c_d=  permute(hd_c, [1 3 2] );  %%% J * C *M
 
       %Neg_Ph  = pagefun( @compute_P_h_all_labels,  h_c_d(1:J,:,:),  hid_yMax(1:J,:),  hidbiasesMax(1:J)' ) ;%%%
       e_dyh = bsxfun( @plus,  bsxfun(@plus,h_c_d(1:J,:,:), hid_yMax(1:J,:)  ) , hidbiasesMax(1:J)'  ) ;
       Neg_Ph  = 1./(1+exp(-e_dyh));
  
  
       P_z_on_v_all_labels =bsxfun(@rdivide,  p_zy_on_v , PP_y_v); %%% J*C*M

       Cdf_z_v_all_y = cumsum(P_z_on_v_all_labels );
 
  
       Neg_phase_for_sum = bsxfun(@times,  PP_y_v,  Neg_Ph(1:J,:,:).*(1- Cdf_z_v_all_y(1:J,:,:)) ); %%% J*C*M

       Neg_phase_for_multiply = squeeze(  sum(Neg_phase_for_sum,  2)  ); %%% J*M
  

       %Neg_phase_W = pagefun( @mtimes, Neg_phase_for_multiply , data );  %%% J*V
       Neg_phase_W = Neg_phase_for_multiply * data ;  %%% J*V
       Neg_phase_hb =sum(  Neg_phase_for_multiply ,2 );%%% J*1
       Neg_phase_U = sum( Neg_phase_for_sum, 3 );
       Neg_phase_yb = sum(P_y_v, 2).';


       if use_gpu
          negprods_dis = zeros(Maxnumhid,numdims,'gpuArray');
          neghidact_dis = zeros(1,Maxnumhid,'gpuArray');
          neg_hidy_dis = zeros(Maxnumhid,numclasses,'gpuArray');
       else
          negprods_dis = zeros(Maxnumhid,numdims);
          neghidact_dis = zeros(1,Maxnumhid);
          neg_hidy_dis = zeros(Maxnumhid,numclasses);
           
       end
  
       negprods_dis( 1:J,: )    = Neg_phase_W ;
       neghidact_dis( 1:J )   = Neg_phase_hb';
       negyact_dis     =  Neg_phase_yb;
       neg_hidy_dis( 1:J,: )    =  Neg_phase_U;
       
%%%%%%%%%%%%% Total gradients %%%%%%%%%%%

      if gen_uselabel
          posprods = (1)* posprods_dis + a* posprods_gen;
          negprods = (1)* negprods_dis + a* negprods_gen;

          poshidact = (1)* poshidact_dis +  a* poshidact_gen;
          neghidact = (1)* neghidact_dis +  a* neghidact_gen;
          
          posvisact =  a* posvisact_gen;
          negvisact =  a* negvisact_gen;
 
          posyact   =  (1)* posyact_dis + a*posyact_gen;
          negyact   =  (1)* negyact_dis + a*negyact_gen ;

          pos_hidy  =  (1)* pos_hidy_dis + a*pos_hidy_gen; 
          neg_hidy  =  (1)* neg_hidy_dis + a*neg_hidy_gen ;
          
      else
          posprods = (1)* posprods_dis + a/(1+a)* posprods_gen;
          negprods = (1)* negprods_dis + a/(1+a)* negprods_gen;
          poshidact = (1)* poshidact_dis +  a/(1+a)* poshidact_gen;
          neghidact = (1)* neghidact_dis +  a/(1+a)* neghidact_gen;
          
          posyact   =  (1)* posyact_dis  ;
          negyact   =  (1)* negyact_dis  ;
          pos_hidy  =  (1)* pos_hidy_dis ; 
          neg_hidy  =  (1)* neg_hidy_dis ; 
          
          posvisact =  a/(1+a)* posvisact_gen;
          negvisact =  a/(1+a)* negvisact_gen;
          
      end
          
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

       if lr_normal
           ordinaryUpDate_L1R ;

       else
           adaptive_lr_update;
       end

       %if max(Pos_numhid)==J+1
       if max(Pos_numhid)==J+1 && max( neg_numhid_gen) == J+1
           %if length(find(Pos_numhid==J+1)) >numcases*0.1;
           %if ~isempty(find(Pos_numhid==J+1, 1))
           J = J+1;
       end

       if batch == round(numbatches/2)
           
           figure(1);
           imagesc(hid_visMax(1:J,:));
           figure(2);
           imagesc(hid_yMax(1:J,:)); 
           figure(3);
           dispims(negdata',28,28) ;
           figure(4);
           dispims(hid_visMax(1:round(J),:)',28,28) ;
           drawnow
           
       end
  
       Maxnegnumhid(batch) = gather( max(neg_numhid_gen) );
       Meannegnumhid(batch) = gather( mean(neg_numhid_gen) );
     
       Maxposnumhid(batch) = gather( M_pnh );
       Meanposnumhid(batch)= gather( mean(Pos_numhid) );
       Minposnumhid(batch) = gather( min(M_pnh) );

   end
   

   mean_Mnegnumhid(epoch) = mean(Maxnegnumhid);
   mean_Mposnumhid(epoch) = mean(Maxposnumhid);
   mean_minposnumhid(epoch) = mean(Minposnumhid); 
   Max_mean_epoch = max(Meannegnumhid);
   mean_mean_epoch(epoch) = mean(Meanposnumhid);
   mean_maxPN_epoch = round( mean_Mposnumhid(epoch) ); 
   min_maxPN_epoch = min(Maxposnumhid); 
  
   fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
  
   numh  = gather (  round( mean_Mposnumhid(epoch) )  );
   
    if use_mom
         momentum = momentum + 0.05; %% momentums for visbiases and ybiases
         momentum = min(0.9 ,momentum);
         
         index_u = find (mom(1:numh)>=0.9); %% momentums for weights and hidden biases
         index_m = find ( mom(1:numh)>=0.8 .* mom(1:numh)<0.9);
         index_d = find (mom(1:numh)<0.8);
         
         mom(index_d) = mom(index_d) + mom_inc;
         mom(index_m) = mom(index_m) + 0.2*mom_inc;
         mom(index_u) = mom(index_u) + 0.1*mom_inc;
         
         mom(1:numh) = min(0.99 ,mom(1:numh));
         
    end
  
   

   h_vMax = gather( hid_visMax(1:J,:) );
   yb = gather( ybiases );
   vb= gather ( visbiases );
   h_yMax = gather( hid_yMax(1:J,:) );
   hbMax = gather ( hidbiasesMax(1:J) );
   if discard
      discard_hids_simple;
      h_vMax = gather( vishid' );
      yb = gather( ybiases );
      vb= gather ( visbiases );
      h_yMax = gather( labhid' );
      hbMax = gather ( hidbiases );
   end
   final  = 0;
   %valid;
   if use_valid
       valid_exact; %%Validation error
   else
       error_exact; %%Training error
   end
   test_epoch(1,epoch) = TestAccuracy;
   %test_epoch(2,epoch) = gather(numh); 
   test_epoch(2,epoch) = eff_nh; 
   Test_midtime;
   T_epoch(epoch) = TA;
  if max_ValAccy <= TestAccuracy && epoch < (M_epoch + stopepochs)
 
     
        max_ValAccy = TestAccuracy;
        M_hid_visMax = gather( h_vMax );
        M_epoch = gather( epoch );
        %M_J = J;
        M_J = length(hbMax);
        M_ybiases = gather( ybiases );
        M_hid_yMax = gather( h_yMax );
        M_hidbiasesMax = gather ( hbMax );
        M_numhid  = gather (  round( mean_Mposnumhid(M_epoch) )  );
        %M_numhid = length(hbMax);
        Max_J_r = gather(Max_J_r);
        save best_Dis_iRBM  M_hid_visMax M_epoch M_hidbiasesMax M_ybiases M_hid_yMax ...
             M_numhid a WC J_r M_J beta0 WH global_lr regularization gen_uselabel use_mom
   %     save parameters_midtime;
        %save parametersZZZZZZZZZZZZ

   elseif max_ValAccy > TestAccuracy && epoch < (M_epoch + stopepochs)
           
          
   else
       break
   end
    
    if use_valid
        figure (5);
        plot(test_epoch(1,1:epoch),'g');
        xlabel('epoch');
        ylabel('Accuracy');
        fprintf(1, 'epoch %4i , maximum number of z %4i , \n validation accuracy %6.4f  \n', epoch, J ,TestAccuracy);
        hold on;
        plot(T_epoch(1:epoch),'r');
        legend('Validation accuracy','Testing accuracy','Location','NorthEastOutside')        
    else
        figure (5);
        plot( (1-test_epoch(1,1:epoch))*1 ,'g' );
        xlabel('epoch');
        ylabel('Accuracy');
        fprintf(1, 'epoch %4i , maximum number of z %4i , \n training accuracy %6.4f  \n', epoch, J ,TestAccuracy);    
        hold on;
        plot((1-T_epoch(1:epoch))*1,'r');
        legend('Training error','Testing error','Location','NorthEastOutside')
    end

    pause(2);
 
end;

