
% Training the iRBM with random permutation of hidden units.
% Code provided by Xuan Peng
% 2016-2017

if restart ==1
    restart=0;
    %%%%%% 超参数/hyper parameters%%%%%
    beta0 = 1.01;WH = 0;  
    epW      = 1;   % Learning rate for weights (old ,now useless, same below)
    epvb     = 1;   % Learning rate for biases of visible units 
    ephb     = 1;   % Learning rate for biases of hidden units 
    
    regularization = 'L1'; %%使用哪种正则化约束/which regularization is chosen
    WC  = 0.0001;
    use_RP = 1;
    h = 1e-10;
    p=1;
    start_lr = 0.05;
    CD= 10;  
    
    global_lr = 0.05;
    Num_inter_initial_lr = 0; %%按照初始学习率迭代的次数。
    initial = Num_inter_initial_lr*ones(1,Maxnumhid);
    start = 1;
    
    lr_normal = 0; %%使用哪种学习率
    lr_adaptive=1; adagrad = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    use_valid = 1;
    makebatches_mnist;
    %makebatches;
    [numcases, numdims, numbatches]=size(batchdata);
    if use_gpu
       batchdata = gpuArray(batchdata);
    end
    
    Maxnumhid = 100;%%%初始容量/initial capacity of oRBM
    order = 0;
    label = 0;
    discard=0;random=1;  
    J=2;
    J_r = 1;
    numhid=0;
    if use_gpu
        weightcost= zeros(Maxnumhid,numdims,'gpuArray');
        hycost=zeros(Maxnumhid,numclasses,'gpuArray'); 
        epsilonW = zeros(Maxnumhid,numdims,'gpuArray');
     %  epsilon_hy = zeros(Maxnumhid,numclasses); 
        epsilonhb  = zeros(1,Maxnumhid,'gpuArray');
     %  epsilonyb  =  zeros(1,numclasses);       
        epsilonvb  = zeros(1,numdims,'gpuArray');

        LzW = zeros(Maxnumhid,numdims,'gpuArray');
        LzU = zeros(Maxnumhid,numclasses,'gpuArray'); 
        LzHb  = zeros(1,Maxnumhid,'gpuArray');
    else
        weightcost= zeros(Maxnumhid,numdims);
        hycost=zeros(Maxnumhid,numclasses); 
        epsilonW = zeros(Maxnumhid,numdims);
     %  epsilon_hy = zeros(Maxnumhid,numclasses); 
        epsilonhb  = zeros(1,Maxnumhid);
     %  epsilonyb  =  zeros(1,numclasses);       
        epsilonvb  = zeros(1,numdims);

        LzW = zeros(Maxnumhid,numdims);
        LzU = zeros(Maxnumhid,numclasses); 
        LzHb  = zeros(1,Maxnumhid);
        
    end
  
    initialmomentum  = 0.0;  
    mom = initialmomentum * ones( Maxnumhid,1 );
    %finalmomentum    = 0.0;
    momentum=initialmomentum;

    lr = 1*ones( Maxnumhid,1 );

    
% Initializing symmetric weights and biases. 
    if use_gpu
        %numclasses =10;
        visbiases     = zeros(1,numdims,'gpuArray');%%%%v的偏置
        %ybiases      = zeros(1,numclasses ,'gpuArray');            %%%标签y的偏置，这里y的维数为10
   
        hid_visMax    = zeros(Maxnumhid,numdims ,'gpuArray');       %%%最大的W矩阵,列数为Maxnumhid
       % hid_yMax      = zeros(Maxnumhid,numclasses ,'gpuArray');    %%%最大的U矩阵，列数为Maxnumhid
        hidbiasesMax  = -0*ones(1,Maxnumhid ,'gpuArray');           %%%h的偏置初始化  
       
        poshidprobs = zeros(numcases,Maxnumhid ,'gpuArray');%%%正阶段h的期望，一次计算一个batch。
        neghidprobs = zeros(numcases,Maxnumhid ,'gpuArray');%%%负阶段h的期望，一次计算一个batch。
        posprods    = zeros(numdims,Maxnumhid ,'gpuArray');
        negprods    = zeros(numdims,Maxnumhid ,'gpuArray');
  
        hid_visMax_inc  = zeros(Maxnumhid,numdims ,'gpuArray');
        hid_yMax_inc    = zeros(Maxnumhid,numclasses ,'gpuArray');
        hidbiasesMaxinc   = zeros(1,Maxnumhid ,'gpuArray');
        ybiases_inc    = zeros(1,numclasses ,'gpuArray');
        visbiases_inc  = zeros(1,numdims ,'gpuArray');
       
        if lr_adaptive
           grad_W_history = zeros(Maxnumhid , numdims ,'gpuArray');
           grad_hb_history = zeros(1,Maxnumhid ,'gpuArray');
           grad_yb_history = zeros(1, numclasses ,'gpuArray');
           grad_U_history = zeros(Maxnumhid, numclasses ,'gpuArray');
           grad_vb_history = zeros(1, numdims ,'gpuArray');
           initial_lr = 0*h;
           W_in_history = zeros(Maxnumhid , numdims ,'gpuArray');
           hb_in_history = zeros(1,Maxnumhid ,'gpuArray');
           yb_in_history = zeros(1, numclasses ,'gpuArray');
           vb_in_history =zeros(1,numdims);
           U_in_history = zeros(Maxnumhid, numclasses ,'gpuArray');
       
           W_in_history(:,:)  = initial_lr^2;
           U_in_history(:,:)  = initial_lr^2;       
           hb_in_history(:) = initial_lr^2;
           yb_in_history(:) = initial_lr^2;
           vb_in_history(:) = initial_lr^2;
        end
    else
        %numclasses =10;
        visbiases     = zeros(1,numdims);%%%%v的偏置
        %ybiases      = zeros(1,numclasses);            %%%标签y的偏置，这里y的维数为10
   
        hid_visMax    = zeros(Maxnumhid,numdims);       %%%最大的W矩阵,列数为Maxnumhid
       % hid_yMax      = zeros(Maxnumhid,numclasses);    %%%最大的U矩阵，列数为Maxnumhid
        hidbiasesMax  = -0*ones(1,Maxnumhid);           %%%h的偏置初始化  
       
        poshidprobs = zeros(numcases,Maxnumhid);%%%正阶段h的期望，一次计算一个batch。
        neghidprobs = zeros(numcases,Maxnumhid);%%%负阶段h的期望，一次计算一个batch。
        posprods    = zeros(numdims,Maxnumhid);
        negprods    = zeros(numdims,Maxnumhid);
  
        hid_visMax_inc  = zeros(Maxnumhid,numdims);
        hid_yMax_inc    = zeros(Maxnumhid,numclasses);
        hidbiasesMaxinc   = zeros(1,Maxnumhid);
        ybiases_inc    = zeros(1,numclasses);
        visbiases_inc  = zeros(1,numdims);
       
        if lr_adaptive
           grad_W_history = zeros(Maxnumhid , numdims);
           grad_hb_history = zeros(1,Maxnumhid);
           grad_yb_history = zeros(1, numclasses);
           grad_U_history = zeros(Maxnumhid, numclasses);
           grad_vb_history = zeros(1, numdims);
           %initial_lr = 0*h;
           W_in_history = zeros(Maxnumhid , numdims);
           hb_in_history = zeros(1,Maxnumhid);
           yb_in_history = zeros(1, numclasses);
           vb_in_history =zeros(1,numdims);
           U_in_history = zeros(Maxnumhid, numclasses);
       
           %W_in_history(:,:)  = initial_lr^2;
           %U_in_history(:,:)  = initial_lr^2;       
           %hb_in_history(:) = initial_lr^2;
           %yb_in_history(:) = initial_lr^2;
           %vb_in_history(:) = initial_lr^2;
        end
        
    end
 
    epoch=1;
    M_epoch = 1;
    minerrsum = 1e100;
    M_LL = -inf;  
    tt  = 1;
    
    %%%start states of the Gibbs chains
    if use_gpu
        negvisprobs = rand(numcases,numdims);
     %   negDstates = gpuArray(negDstates);
        negdata = 0.5 > negvisprobs;
        negdata = gpuArray(negdata);
        negdata = round(negdata);
        neg_numhid_gen = ones(1,numcases,'gpuArray');
    else
        negvisprobs = rand(numcases,numdims);
        negdata = 0.5 > negvisprobs;
        negdata = round(negdata);
        neg_numhid_gen = ones(1,numcases);
    end
end

 
for epoch = epoch:maxepoch
       
     makebatches_mnist; 
     %makebatches;
     [numcases numdims numbatches]=size(batchdata);
     if use_gpu
         batchdata = gpuArray(batchdata);
%         batchtargets = gpuArray(batchtargets);
     end
    
     if lr_normal
        %learning_rate = 0.5/ceil(epoch/50); %%% learning rate decay
        learning_rate = 0.5/( epoch/50+1 );
        learning_rate = max( learning_rate , 0.001);
        epW      = learning_rate;   % Learning rate for weights 
        ephy      =  learning_rate;   %y与h权值矩阵的学习率

        ephb       = learning_rate;   % Learning rate for biases of hidden units 
        epyb       = learning_rate;   %标签y的偏置的学习率
        epvb       = learning_rate;
    
     end
     errsum=0;
%       if epoch >1
%          train =1; discard = 1;
%          discard_hids;

     Maxnegnumhid = zeros(1,numbatches);
     Meannegnumhid = zeros(1,numbatches); 
     Maxposnumhid = zeros(1,numbatches);
     Minposnumhid = zeros(1,numbatches);


  for batch = 1:numbatches
     
        %train =1;
        %discard_hids;
        
     if J >= Maxnumhid -1   %%%%增大“容量”/grow the "capacity" of oRBM if necessary
        Maxnumhid = Maxnumhid + 100;
        
        
        hid_visMax(J+1:Maxnumhid,:)      = 0; 
        hid_yMax (J+1:Maxnumhid,:)       = 0;
        hidbiasesMax(J+1:Maxnumhid)      = 0;  
        hid_visMax_inc(J+1:Maxnumhid,:)  = 0;
        hid_yMax_inc(J+1:Maxnumhid,:)    = 0;
        hidbiasesMaxinc(J+1:Maxnumhid)   = 0;
        
        epsilonW(J+1:Maxnumhid,:) = 0;
        epsilon_hy(J+1:Maxnumhid,:) = 0; 
        epsilonhb(J+1:Maxnumhid)  = 0;
        
        initial(J+1:Maxnumhid) = Num_inter_initial_lr;
        
        if lr_adaptive
           grad_W_history(J+1:Maxnumhid,:) = 0;
           grad_U_history(J+1:Maxnumhid,:) = 0;
           grad_hb_history(J+1:Maxnumhid)= 0;
         
           W_in_history(J+1:Maxnumhid,:) =  initial_lr^2;
           U_in_history(J+1:Maxnumhid,:)=  initial_lr^2;
           hb_in_history(J+1:Maxnumhid)=  initial_lr^2;
           
           mom(J+1:Maxnumhid)= initialmomentum;
           lr(J+1:Maxnumhid)= 1;
        end
        if use_gpu
            weightcost= zeros(Maxnumhid,numdims,'gpuArray');
            hycost=zeros(Maxnumhid,numclasses,'gpuArray'); 
        
            LzW = zeros(Maxnumhid,numdims ,'gpuArray');
            LzU = zeros(Maxnumhid,numclasses ,'gpuArray'); 
            LzHb  = zeros(1,Maxnumhid ,'gpuArray');
        else
            weightcost= zeros(Maxnumhid,numdims);
            hycost=zeros(Maxnumhid,numclasses); 
        
            LzW = zeros(Maxnumhid,numdims);
            LzU = zeros(Maxnumhid,numclasses); 
            LzHb  = zeros(1,Maxnumhid);
        end

        
     end
    
        if use_RP
              random_order_simple;
        end  
        
       if use_gpu
           lwc = gpuArray.linspace (1,1, J )';
           lr = gpuArray.linspace(1,1, Maxnumhid )';
           bt = linspace(1,1, Maxnumhid );
       else
           lwc = linspace (1,1, J )';
           lr = linspace(1,1, Maxnumhid )';
           bt = linspace(1,1, Maxnumhid );
           
       end

     epsilonW     = epW .* repmat( lr ,1, numdims );   % Learning rate for weights 
   %  epsilon_hy     =  ephy.* repmat( lr ,1,numclasses );   %y与h权值矩阵的学习率
     epsilonhb       = ephb.* lr';   % Learning rate for biases of hidden units 
   %  epsilonyb       =epyb;   %标签y的偏置的学习率
     epsilonvb       = epvb;
     weightcost(1:J,:) =  WC* repmat( lwc ,1, numdims );
     hycost(1:J,:) =  WC* repmat( lwc , 1, numclasses );
     %fprintf(1,'epoch %d batch %d\r',epoch,batch); 
     fprintf(1,'epoch %d batch %d z %d \r',epoch, batch, J); 

     visbias = repmat(visbiases,numcases,1);

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   data = data > rand(numcases,numdims,'gpuArray');  
    data = ( batchdata(:,:,batch));
  %%%正过程，从p(z|v)中采样得到z，若z>J0=10,z=J0+1;%%% 注意！权值要乘2以补偿上层向下的权值！
    hidbias = 1*hidbiasesMax;
    beta = beta0 *bt.* soft_plus(WH * hidbias );  
    P_z_on_v = P_z( data , hid_visMax ,hidbiasesMax , J , beta ,beta0 ,numcases ); %%%compute P(z|v) 
    sum_P_z_on_v = cumsum(P_z_on_v);
    [Pos_numhid_mask, Pos_numhid_1hot ] = Sample_z (P_z_on_v,numcases,J); %%% sample z from P(z|v)
    [~,Pos_numhid_gen] = max(Pos_numhid_1hot);
    where_Jpp = find(Pos_numhid_gen==J+1);
    Pos_numhid_1 = Pos_numhid_gen;
    Pos_numhid_1(where_Jpp)=1;
   %M_Pnh=max(Pos_numhid_1);
   
    [~,M_Pnhs] = max(P_z_on_v);
    M_Pnh = max(M_Pnhs);
   %M_Pnh = max(Pos_numhid_gen);
   %%%采样完毕%%%
    if use_gpu
        S_PzOnvy = zeros(Maxnumhid,numcases,'gpuArray');
        Pos_MaxNh =zeros(Maxnumhid,numcases,'gpuArray');
        Pos_numhid4 =zeros(Maxnumhid,numcases,'gpuArray');
    else 
        S_PzOnvy = zeros(Maxnumhid,numcases);
        Pos_MaxNh =zeros(Maxnumhid,numcases);
        Pos_numhid4 =zeros(Maxnumhid,numcases);
        
    end
    
    S_PzOnvy(1:J,:) = sum_P_z_on_v(1:J,:);
    Pos_MaxNh(1:J,:)= 1;
    Pos_numhid4(1:J+1,:)= Pos_numhid_mask;
    if use_gpu
        poshidprobs = 1./(   1 + exp( bsxfun( @minus, pagefun( @mtimes, -1*hid_visMax , data' ) , hidbias') )  );
    else
        poshidprobs = 1./(1 + exp(  -1*hid_visMax * data'  -repmat( hidbias',1,numcases )));
    end
 
    Dbeta_h =    1 * WH * beta0 * exp(WH * repmat( hidbias',1,numcases ) )./(1+ exp(WH * repmat( hidbias',1,numcases ) ));
    poshidprobsMinusDbeta = poshidprobs-  Dbeta_h;
   % poshidprobsMinusDbeta = poshidprobsMinusDbeta.* Pos_numhid4;
    poshidprobsMinusDbeta = poshidprobsMinusDbeta.* Pos_MaxNh;
   % poshidprobs= poshidprobs.* Pos_numhid4;
    poshidprobs= poshidprobs.*  Pos_MaxNh;
%    poshidprobs = gather(poshidprobs);
%    posprods    = poshidprobs * data ;
%    posprods    = gpuArray(posprods);   
%    posprods    = pagefun(@mtimes, poshidprobs , data ); 
  %  poshidact   = sum(poshidprobsMinusDbeta , 2).';
    %posprods    = pagefun(@mtimes, poshidprobs.*( 1-  S_PzOnvy ) , data ); 
    posprods    = poshidprobs.*( 1-  S_PzOnvy ) * data   ; %%%用GPU时可能会报错
    poshidact   = sum(poshidprobsMinusDbeta.*( 1-  S_PzOnvy ) , 2).'; 
    posvisact = sum(data);
%    posvisact = gpuArray(posvisact);
   
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %negdata = batchdata(:,:,batch);
%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for cditer=1:CD
   %%%%%%% z~p(z|v)%%%%%%%%%%%%%
   
        P_z_on_v_neg = P_z( negdata , hid_visMax ,hidbiasesMax , J , beta ,beta0, numcases ); 
        sum_P_z_on_v_neg = cumsum(P_z_on_v_neg);
        [Neg_numhidmask, Neg_numhid1hot] = Sample_z (P_z_on_v_neg,numcases,J);
        [~,neg_numhid_gen] = max(Neg_numhid1hot); 
        where_NJpp = find(neg_numhid_gen==J+1);
        neg_numhid_1 = neg_numhid_gen;
        neg_numhid_1(where_NJpp)=1;
        M_nnh = max(neg_numhid_1);
       % M_nnh = max(neg_numhid_gen);

   %%%%%%% h~p(h|v,z)%%%%%%%%%%%%% 
       if use_gpu                   
           neg_numhid4 =zeros(Maxnumhid,numcases,'gpuArray');
       else
           neg_numhid4 =zeros(Maxnumhid,numcases);
       end
       neg_numhid4(1:J+1,:)= Neg_numhidmask;
       neghidprobs = 1./(  1 + exp( bsxfun(@minus, -1* hid_visMax* negdata' , hidbias' ) )  ); %%%用GPU时可能会报错   
       %neghidprobs =  1./(   1 + exp( bsxfun( @minus, pagefun( @mtimes, -1*hid_visMax , negdata' ) , hidbias') )  );
  
       neghidprobs = neghidprobs.* neg_numhid4;
       if use_gpu
           neghidstates =  neghidprobs > rand(Maxnumhid ,numcases,'gpuArray');
       else
           neghidstates =  neghidprobs > rand(Maxnumhid ,numcases);
       end
       neghidstates = round(neghidstates);
       %neghidstates = gather(neghidstates);
       
   %%%%%%% v~p(v|h,z)%%%%%%%%%%%%% 权值不用乘2了，去掉了和标签有关的项
       negvisprobs =   1./(   1 + exp( bsxfun(@minus,- hid_visMax'* neghidstates , visbiases' ) )  ); %%%用GPU时可能会报错        
       %negvisprobs =   1./(1 + exp(  pagefun(@mtimes, - hid_visMax', neghidstates ) - repmat( visbiases',1,numcases ))   );   
       if use_gpu
           negdata = negvisprobs > rand(numdims , numcases, 'gpuArray');
       else
           negdata = negvisprobs > rand(numdims , numcases);
       end
       negdata = negdata.';
       negdata = round(negdata);%%%GPU不认逻辑变量！

    end
    if use_gpu
        neg_MaxNh =zeros(Maxnumhid,numcases,'gpuArray');
        s_negPzONvy = zeros(Maxnumhid,numcases,'gpuArray');
    else
        neg_MaxNh =zeros(Maxnumhid,numcases);
        s_negPzONvy = zeros(Maxnumhid,numcases);
    end
  
    neg_MaxNh(1:J,:)= 1;
    s_negPzONvy(1:J,:) = sum_P_z_on_v_neg(1:J,:);
%    negdata = gather(negdata); 
    neghidprobs = 1./(   1 + exp( bsxfun(@minus, -1* hid_visMax* negdata' , hidbias') )   );   %%%用GPU时可能会报错   
    %neghidprobs =  1./(   1 + exp( bsxfun( @minus, pagefun( @mtimes, -1*hid_visMax , negdata' ) , hidbias') )  );
   
   
  %  neghidprobs = neghidprobs.* neg_numhid4;
    neghidprobs = neghidprobs.* neg_MaxNh;
  
    Dbeta_h =    1 * WH * beta0 * exp(WH * repmat( hidbias',1,numcases ) )./(1+ exp(WH * repmat( hidbias',1,numcases ) ));
    neghidprobsMinusDbeta = neghidprobs-  Dbeta_h;
  %  neghidprobsMinusDbeta = neghidprobsMinusDbeta.* neg_numhid4;
   % neghidprobs = neghidprobs.* neg_numhid4; 
  
    neghidprobsMinusDbeta = neghidprobsMinusDbeta.* neg_MaxNh;

 %   neghidprobs = gather(neghidprobs);
 %   negprods  = neghidprobs * negdata;
 %   negprods  = gpuArray(negprods);
 
 
 %   negprods  = pagefun(@mtimes, neghidprobs , negdata );
 %   neghidact = sum(neghidprobsMinusDbeta,2).';
    negvisact = sum(negdata); 
  %  negvisact = gpuArray(negvisact);
  
    %negprods  = pagefun(@mtimes, neghidprobs.*( 1- s_negPzONvy ) , negdata );
    negprods  = neghidprobs.*( 1- s_negPzONvy ) * negdata;%%%%%%用GPU时可能会报错  
    neghidact =  sum(neghidprobsMinusDbeta.*( 1- s_negPzONvy ) ,2).';
 %  negvisact = sum(negdata); 
  
%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err= sum(sum( (data-negdata).^2 ));
    errsum = err + errsum;  

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

       if lr_normal
           ordinary_update ;

       else
           adaptive_lr_update;

       end
  
    start = 0;
%     %if M_Pnh ==J+1;
       %if length(find(Pos_numhid_gen==J+1))>numcases/5
       if ~isempty( find( (Pos_numhid_gen==J+1).*(neg_numhid_gen==J+1), 1 ) )
          J = J+1;

       end



%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     if batch == round(numbatches/2)
       figure(3);   
       imagesc(hid_visMax(1:J,:));    
       figure(4); 
       dispims(negdata',28,28);
       drawnow
     end
     neg_numhid_gen = gather(neg_numhid_gen);
     Maxposnumhid(batch) = gather( M_Pnh );
     Maxnegnumhid(batch) = max(neg_numhid_gen);
     Meannegnumhid(batch) = mean(neg_numhid_gen);
     Minposnumhid(batch) = gather( min(Pos_numhid_gen) );
     
      
   end
   mean_Mnegnumhid(epoch) = mean(Maxnegnumhid);%%%逼死强迫症
   Max_mean_new = max(Meannegnumhid);
   mean_mean_epoch(epoch) = mean(Meannegnumhid);
   mean_Mposnumhid(epoch) = mean(Maxposnumhid);
   mean_minposnumhid(epoch) = mean(Minposnumhid); 
   mean_maxPN_epoch = round( mean_Mposnumhid(epoch) ); 
   min_maxPN_epoch = min(Maxposnumhid); 

   fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 

   if rem (epoch,50)==0
       %numhid = round(  min( mean_Mnegnumhid(epoch-9:epoch) ) ); 
       %numhid = round(  min( mean_Mposnumhid(epoch-9:epoch) ) ); 
       numh = gather (round(  mean_Mposnumhid(epoch) )) ;
       vh = gather( hid_visMax(1:numh , :).'); 
       hb = gather( hidbiasesMax(1:numh));  
       vb = gather(visbiases);
%      train = 0;
%      discard_hids
       save fullmnistvh vh vb hb epoch numh errsum J_r beta0
       if use_gpu==0
           batchdata = gather(batchdata);
       end      
       rbm_AIS_estimate %%% Convert the model into an RBM, and compute the LL on the validation set. 
       %iRBM_AIS_estimate
       vh = gather( vh ); 
       hb = gather( hb );  
       vb = gather( vb );
       if loglik_test_est > M_LL
          M_LL = gather(loglik_test_est);
          errs = gather(errsum);
          save best_iRBM vh vb hb epoch numh errs M_LL J_r beta0
       end
       LL_history(1,tt) = loglik_test_est;
       LL_history(2,tt) = numh;
       tt = tt+1;
   end
      
%  if rem(epoch,100)==0
%     save parameters_midtime_irbm
%  end

      
end;



