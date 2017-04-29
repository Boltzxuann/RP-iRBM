
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


grad_W  = (posprods-negprods)/numcases -  Wcost;
grad_hb = (poshidact-neghidact)/numcases   ;
grad_vb =  (posvisact-negvisact)/numcases  ;
if label ==1
      grad_yb =  (posyact-negyact)/numcases   ;
      grad_U  = (pos_hidy(:,1:numclasses )-neg_hidy(:,1:numclasses ))/numcases  - Ucost;
end

hid_visMax_inc = bsxfun( @times, mom, hid_visMax_inc )  + epsilonW.*( grad_W ); 
visbiases_inc = momentum*visbiases_inc +  (epsilonvb).*( grad_vb );
hidbiasesMaxinc = mom' .* hidbiasesMaxinc + (epsilonhb).*( grad_hb );      
if label ==1
       hid_yMax_inc(:,1:numclasses )  = bsxfun( @times, mom, hid_yMax_inc(:,1:numclasses ) ) + (epsilon_hy).* ( grad_U );    
       ybiases_inc(1:numclasses ) = momentum * ybiases_inc(1:numclasses ) + (epsilonyb).*( grad_yb );    
end
     
%%%% updates
hid_visMax = hid_visMax + hid_visMax_inc;
hidbiasesMax = hidbiasesMax + hidbiasesMaxinc;
visbiases = visbiases + visbiases_inc;
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
% if  ~isempty( find (abs(hidbiasesMax) >maxnorm , 1 ))
%         hb_onoff =  abs(hbMax)>maxnorm ;
%         index_hb = find( hb_onoff );
%         hidbiasesMax(index_hb) = hidbiasesMax(index_hb) .* maxnorm./ abs(hidbiasesMax(index_hb));
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


  
