

hidvis1 = hid_visMax(1:J,:);
norm_hidvis = sqrt(  sum (hidvis1.^2, 2)  ) ;
mean_norm = mean( norm_hidvis );
if order ==0
   if train ==1 
    hids_onoff = norm_hidvis >  0.3* mean_norm;
   else
     hids_onoff = norm_hidvis > 0.1* mean_norm;
   end
  index= find(hids_onoff);
 
  if discard
      J1 = length(index); %%%剩下的长度
      J=J1;%%% 这个操作使得J缩小了！
      hidvis_last = hidvis1 (index,:);
      if label == 1
         hidy_last = hid_yMax(index,:);
      end
      hidbiases_last = hidbiasesMax(index);
  else
      J1 = max(index);
      
  end



  if train ==1
      if discard   
          
        grad_W_history(1:J,:) = grad_W_history(index,:);%%%先将剩下的紧凑的排在一起
        grad_W_history(J+1:Maxnumhid,:) = 0;
    
        grad_U_history(1:J,:) = grad_U_history(index,:);
        grad_U_history(J+1:Maxnumhid,:) = 0;
    
        grad_hb_history(1:J) = grad_hb_history(index);
        grad_hb_history(J+1:Maxnumhid) = 0; 
        
        W_in_history(1:J,:) =  W_in_history(index,:);
        W_in_history(J+1:Maxnumhid,:) = 0;
        
        U_in_history(1:J,:) =  U_in_history(index,:);
        U_in_history(J+1:Maxnumhid,:) = 0;
        
        hb_in_history(1:J)=  hb_in_history(index);
        hb_in_history(J+1:Maxnumhid) = 0;
        
      end
      if random
         index2=randperm(J1);%%%打乱顺序，这里的J可能是index的最后一个（如果discard为0），也可能是index的长度（如果discard大于0）
         grad_W_history(1:J1,:) = grad_W_history(index2,:);
         grad_U_history(1:J1,:) = grad_U_history(index2,:);
         grad_hb_history(1:J1) = grad_hb_history(index2);
         
         W_in_history(1:J1,:) =  W_in_history(index2,:);
         U_in_history(1:J1,:) =  U_in_history(index2,:);
         hb_in_history(1:J1)=  hb_in_history(index2);
         
         
      end
  end



end
   
   
   if order ==1
[norm_hid_descend, index_de] = sort(norm_hidvis,'descend');

hidvis_descend = hidvis1 (index_de,:);
if label ==1
hidy_descend = hid_yMax(index_de,:);
end
hidbiases_descend = hidbiasesMax(index_de);
 grad_W_history  = grad_W_history(index_de,:);
 grad_U_history  = grad_U_history(index_de,:);
 grad_hb_history = grad_hb_history(index_de);
 
 W_in_history  = W_in_history(index_de,:);
 U_in_history  = U_in_history(index_de,:);
 hb_in_history = hb_in_history(index_de);
if train == 1
    hids_onoff2 = norm_hid_descend> 0.1* mean_norm;
else
    hids_onoff2 = norm_hid_descend> 0.1* mean_norm;
end

index_de_last = find(hids_onoff2);

hidvis_descend_last = hidvis_descend (index_de_last,:);
   
if train ==1
    J = size(hidvis_descend_last,1);
    %J = max(index_de_last);
    grad_W_history(1:J,:)  = grad_W_history(index_de_last,:);
    grad_W_history(J+1:Maxnumhid,:) = 0;
    
    grad_U_history(1:J,:) = grad_U_history(index_de_last,:);
    grad_U_history(J+1:Maxnumhid,:) = 0;
    
    grad_hb_history(1:J) = grad_hb_history(index_de_last);
    grad_hb_history(J+1:Maxnumhid) = 0;
    
        W_in_history(1:J,:) =  W_in_history(index_de_last,:);
        W_in_history(J+1:Maxnumhid,:) = 0;
        
        U_in_history(1:J,:) =  U_in_history(index_de_last,:);
        U_in_history(J+1:Maxnumhid,:) = 0;
        
        hb_in_history(1:J)=  hb_in_history(index_de_last);
        hb_in_history(J+1:Maxnumhid) = 0;
    
end


if label == 1
    hidy_descend_last = hidy_descend(index_de_last,:);
end
hidbiases_descend_last = hidbiases_descend(index_de_last);


   end

if order == 1
    if label == 1
        labhid = hidy_descend_last.'; labbiases = ybiases; vishid = hidvis_descend_last.'; hidbiases = hidbiases_descend_last; 
        if train ==1
          
           hid_yMax(1:J,:) = hidy_descend_last; hid_visMax(1:J,:)= hidvis_descend_last;hidbiases(1:J) = hidbiases_descend_last;
           
        
        end
        
    else
         vishid = hidvis_descend_last.'; hidbiases = hidbiases_descend_last;  
         if train == 1
             hid_visMax(1:J,:)= hidvis_descend_last;hidbiases(1:J) = hidbiases_descend_last;
         end
    end
else
    if label ==1
        %labhid = hidy_last.'; labbiases = ybiases; vishid = hidvis_last.'; hidbiases = hidbiases_last; 
         labhid = hid_yMax(1:J1,:).'; labbiases = ybiases; vishid = hid_visMax(1:J1,:).'; hidbiases = hidbiasesMax(1:J1); 
        if train ==1 %%%只有当训练的时候，才打乱原本的顺序。
           if discard
           hid_yMax(1:J1,:) = hidy_last; hid_visMax(1:J,:)= hidvis_last;hidbiasesMax(1:J1) = hidbiases_last;
           end
           if random
               hid_yMax(1:J1,:) = hid_yMax(index2,:);hid_visMax(1:J1,:)=hid_visMax(index2,:);hidbiasesMax(1:J1)=hidbiasesMax(index2);
               
           end
        
        end
        

     
    else
       % vishid = hidvis_last.'; hidbiases = hidbiases_last;；
         vishid = hid_visMax(1:J1,:).'; hidbiases = hidbiasesMax(1:J1); %%%J1要么是L2norm大于门限值的隐藏单元的最大序号或是个数。
         if train ==1
                if discard
                hid_visMax(1:J1,:)= hidvis_last;hidbiasesMax(1:J1) = hidbiases_last;
                end
            if random
              hid_visMax(1:J1,:)=hid_visMax(index2,:);hidbiasesMax(1:J1)=hidbiasesMax(index2);
               
            end
        
        end
        
        
        
    end
end

    