%%%%%%%%% adaptive learning rate updates (adagrad and adadelta ) %%%%%%

n_epoch = 1;
if adagrad == 0
    if strcmp(regularization , 'no')
        Wcost = 0;
        Ucost = 0;
    elseif strcmp(regularization , 'L1')
        Wcost = weightcost.*sign(hid_visMax);     
        Ucost = hycost.*sign(hid_yMax);
    elseif strcmp(regularization,'L2') 
        Wcost = weightcost.*hid_visMax;
        Ucost = hycost.* hid_yMax;
    else
        Wcost = 0;
        Ucost = 0;
    end
    
   grad_W  = (posprods-negprods)/numcases  -  Wcost;
   grad_hb = (poshidact-neghidact)/numcases  ;
   grad_vb =  (posvisact-negvisact)/numcases  ;
   if label == 1
       grad_U  = ( pos_hidy(:,1:numclasses )-neg_hidy(:,1:numclasses ) )/numcases  - Ucost;
       grad_yb =  (posyact-negyact)/numcases   ;
   end
   
   
else
    if strcmp(regularization , 'no')
        Wcost = 0;
        Ucost = 0;
    elseif strcmp(regularization , 'L1')
        Wcost = global_lr.* epsilonW .* weightcost.*sign(hid_visMax);
        %Wcost =  weightcost.*sign(hid_visMax);
        if label == 1
            Ucost = global_lr .* (epsilon_hy).* hycost.*sign(hid_yMax(:,1:numclasses ));
            %Ucost = hycost.*sign(hid_yMax(:,1:numclasses ));
        end
        
    elseif strcmp(regularization,'L2') 
        Wcost = global_lr .* epsilonW .* weightcost.*(hid_visMax);
        %Wcost =  weightcost.*(hid_visMax);
        if label == 1
            Ucost = global_lr .* (epsilon_hy).* hycost.*(hid_yMax(:,1:numclasses ));
            %Ucost =  hycost.*(hid_yMax(:,1:numclasses ));
        end
    else
        Wcost = 0;
        Ucost = 0;
    end
       

   grad_W  = (posprods-negprods)/numcases; %- Wcost ;
   grad_hb = (poshidact-neghidact)/numcases  ;
   grad_vb =  (posvisact-negvisact)/numcases  ;
   if label ==1
        grad_yb =  (posyact-negyact)/numcases   ;
        grad_U  = ( pos_hidy(:,1:numclasses )-neg_hidy(:,1:numclasses ) )/numcases; %- Ucost ;
   end
   
end

if adagrad    
   grad_W_history = 1*grad_W_history + ( grad_W ).^2;
   grad_hb_history = 1*grad_hb_history + ( grad_hb ).^2;  
   grad_vb_history = 1*grad_vb_history +  (  grad_vb ).^2;
   if label == 1
      grad_yb_history = 1*grad_yb_history + ( grad_yb  ).^2;
      grad_U_history = 1*grad_U_history +  ( grad_U ).^2;
   end
   
   
   n_W = global_lr./ ( sqrt(grad_W_history) + h );%%%¿ÉÄÜ²»¶Ô
   n_hb = global_lr./ ( sqrt(grad_hb_history) + h );
   %n_hb = start_lr;
    
    if start >0
        n_vb = start_lr;
        start= start-1;
    else
        n_vb = global_lr./ ( sqrt(grad_vb_history) + h );
    end
    
    if label == 1
        
        if start >0
            n_yb = start_lr ;
            start= start-1;
        else
            n_yb = global_lr./( sqrt(grad_yb_history) + h );
        end
        
        n_U  = global_lr./ ( sqrt(grad_U_history) + h  );
    end
    index_initial = find(initial(1:J));
    if ~isempty (index_initial)
        n_W(index_initial,:)= start_lr;
        n_U(index_initial,:)= start_lr;
        n_hb(index_initial)= start_lr;
        initial(index_initial)= initial(index_initial)-1;
    end 
else %%% If using adadelta
    grad_W_history = p*grad_W_history + (1)*(grad_W).^2;
    grad_hb_history = p*grad_hb_history + (1)*(grad_hb).^2;

    grad_vb_history = p*grad_vb_history +  (1)*(grad_vb).^2;
    if label  
        grad_yb_history = p*grad_yb_history + (1)*(grad_yb).^2;
        grad_U_history = p*grad_U_history + (1)*(grad_U).^2;
    end

    n_W = ( sqrt(W_in_history ) + 0)./( sqrt(grad_W_history ) + h );%%%¿ÉÄÜ²»¶Ô
    n_hb = ( sqrt(hb_in_history ) + 0)./ ( sqrt(grad_hb_history ) + h);
    %n_hb = start_lr;

    if start >0
        n_vb = start_lr;
        start= start-1;
    else
        n_vb = ( sqrt(vb_in_history) + 0  )./( sqrt(grad_vb_history) + h  ) ; 
    end
    
    if label == 1
        
        if start >0
            n_yb = start_lr ;
            start= start-1;
        else
            n_yb =  ( sqrt(yb_in_history) + 0 )./ ( sqrt(grad_yb_history) + h ) ;
        end
        n_U =  ( sqrt(U_in_history)  + 0 )./ ( sqrt(grad_U_history)  + h );
    end
    index_initial = find(initial(1:J));
    if ~isempty (index_initial)
        n_W(index_initial,:)= start_lr;
        n_U(index_initial,:)= start_lr;
        n_hb(index_initial)= start_lr;
        initial(index_initial)= initial(index_initial)-1;
    end  
end

hid_visMax_inc = bsxfun( @times, mom, hid_visMax_inc )  +  n_W .*epsilonW .*( grad_W ) -  (adagrad==1)*Wcost;  
    %vishid_p_inc = momentum*vishid_p_inc + ...
                %epsilonW_p*( (posprods_2-negprods_2)/numcases - weightcost*vishid_p);
    %visbias_m_inc = momentum*visbias_m_inc + (epsilonvb_m/numcases)*(posvisact-negvisact);
    %visbias_p_inc = momentum*visbias_p_inc + (epsilonvb_p/numcases)*(posvisact_2-negvisact_2);
   
hidbiasesMaxinc = mom' .* hidbiasesMaxinc +  n_hb .*(epsilonhb).*( grad_hb ); 
visbiases_inc = momentum * visbiases_inc +  (epsilonvb).* n_vb.*(grad_vb);    
    %hid_yMax_inc  = momentum* hid_yMax_inc +  n_epoch*(epsilon_hy)* ((pos_hidy-neg_hidy)/numcases-weightcost*hid_yMax) ;
if label ==1
       hid_yMax_inc  = bsxfun( @times, mom, hid_yMax_inc ) +  n_U  .*(epsilon_hy).* ( grad_U ) - (adagrad==1)*Ucost;   
       ybiases_inc = momentum * ybiases_inc +  n_yb .*(epsilonyb).*( grad_yb );   
end

if adagrad == 0 %%% If using adadelta
    W_in_history = p*W_in_history + (1)*hid_visMax_inc.^2;
    hb_in_history = p*hb_in_history + (1)*hidbiasesMaxinc.^2;
    yb_in_history = p*yb_in_history + (1)*ybiases_inc.^2;
    U_in_history = p* U_in_history + (1)*hid_yMax_inc.^2;
    vb_in_history = p*vb_in_history + (1)*visbiases_inc.^2;
end     
        
     
hid_visMax = hid_visMax + hid_visMax_inc;
visbiases = visbiases + visbiases_inc;
hidbiasesMax = hidbiasesMax + hidbiasesMaxinc;
if label == 1
      ybiases = ybiases + ybiases_inc;
      hid_yMax   = hid_yMax   + hid_yMax_inc;
end


%%% maxnorm 
%Constraining weights in a sphere.
maxnorm = 10;
%     maxnorm = gpuArray(maxnorm);
hidvis1 = hid_visMax(1:J,:);
norm_hidvis = sqrt(  sum (hidvis1.^2, 2)  ) ;
%norm_hidvis = gather(norm_hidvis);
if ~isempty( find ( norm_hidvis > maxnorm, 1 ) )
       W_onoff = norm_hidvis > maxnorm;
       index_W = find ( W_onoff  ) ;
       hid_visMax(index_W,:) = hid_visMax(index_W,:).* maxnorm./repmat( norm_hidvis(index_W),1,numdims );
end
%hbMax = gather(hidbiasesMax)   ;  
% maxnormH = 10;
% hbMax = hidbiasesMax(1:J)   ;   
% if  ~isempty( find (abs(hidbiasesMax) >maxnormH , 1 ))
%          hb_onoff =  abs(hbMax)>maxnormH ;
%          index_hb = find( hb_onoff );
%          hidbiasesMax(index_hb) = hidbiasesMax(index_hb) .* maxnormH./ abs(hidbiasesMax(index_hb));
% end
if label == 1  
       maxnormU = 5;
       hidy1 = hid_yMax(1:J,:);
       norm_hidy = sqrt(  sum (hidy1.^2, 2)  ) ;
       %norm_hidy = gather(norm_hidy);
       if ~isempty( find ( norm_hidy > maxnormU, 1 ) )
         U_onoff = norm_hidy > maxnormU;
         index_U = find ( U_onoff  ) ;
         hid_yMax(index_U,:) = hid_yMax(index_U,:).* maxnormU./repmat( norm_hidy(index_U),1,numclasses );
       end
end

%  Constraining weights in a cube       
% index_Wp = find( (hid_visMax) > 5 );
% hid_visMax(index_Wp) = 5; 
% index_Wn = find( (hid_visMax) < -5 );
% hid_visMax(index_Wn) = -5;
% 
% index_Up = find( (hid_yMax) > 5 );
% hid_yMax(index_Up) = 5;
% index_Un = find( (hid_yMax) < -5 );
% hid_yMax(index_Un) = -5;    
