
% Random permutation of hidden units and their corresponding parameters.
later =11;
if  epoch == later && batch == 1
    hidvis1 = hid_visMax(1:J,:);
    norm_hidvis = sqrt(  sum (hidvis1.^2, 2)  ) ;
    mean_norm = mean( norm_hidvis );
    [norm_hid_descend, index_de] = sort(norm_hidvis,'descend');
    
    hidbiasesMax(1:J) = hidbiasesMax(index_de);
    hid_visMax(1:J,:) = hidvis1 (index_de,:);
    
    hidbiasesMaxinc(1:J) = hidbiasesMaxinc(index_de);
    hid_visMax_inc(1:J,:) = hid_visMax_inc(index_de,:);
    
    if label == 1
        hid_yMax(1:J,:) = hid_yMax(index_de,:);
        hid_yMax_inc(1:J,:) = hid_yMax_inc(index_de,:);  
    end

    
    mom(1:J) = mom(index_de);
    initial(1:J)= initial(index_de);
    
       grad_W_history(1:J,:)  = grad_W_history(index_de,:);
       grad_U_history(1:J,:)  = grad_U_history(index_de,:);
       grad_hb_history(1:J) = grad_hb_history(index_de);
       W_in_history(1:J,:)  = W_in_history(index_de,:);
       U_in_history(1:J,:)  = U_in_history(index_de,:);
       hb_in_history(1:J) =  hb_in_history(index_de);

end

%%Set the regrouping rate M_t
    if epoch < 1
        J_r = 1;
    else
        if epoch < later
            J_r = round(0.9*J);
            %J_r = round(  mean( mean_mean_epoch(epoch-1) ) );
        else
            if epoch == later
                J_r = 1;
            else
                %J_r = round(0.8 * mean( mean_Mposnumhid(11:epoch-1) ));
                 if mean_mean_epoch(epoch-1) < 100  
                     J_r = round(  0.9*mean( mean_mean_epoch(epoch-10 :epoch-1) ) );
                 else 
                     J_r =  round( mean( mean_mean_epoch(later :epoch-1) ) )-10 ;
                     if epoch > later+10
                         J_r =  min(  round(  mean( mean_mean_epoch( round( epoch*0.8 ):epoch-1) ) )-10 ,...
                                       round(0.9*J) );  %%% Encourage a safe distance between J and avarage activate number.
                     end                     
                 end
                %J_r = round( ( mean_mean_epoch(epoch-1) ) )-10;
            end          
            %J_r = length(find(norm_hidvis>mean_norm/2));
        end

    end
   
J_r = gather(J_r);
index= randperm(J_r);%% Reorder the hidden units
mom(1:J_r) = mom(index);
initial(1:J_r)= initial(index);
if lr_adaptive
            
             grad_W_history(1:J_r,:) = grad_W_history(index,:);
             grad_U_history(1:J_r,:) = grad_U_history(index,:);
             grad_hb_history(1:J_r) = grad_hb_history(index);
             W_in_history(1:J_r,:) =  W_in_history(index,:);
             U_in_history(1:J_r,:) =  U_in_history(index,:);
             hb_in_history(1:J_r)=  hb_in_history(index);
         
end

hidbiasesMax(1:J_r) = hidbiasesMax(index);
hid_visMax(1:J_r,:) = hid_visMax (index,:);

hidbiasesMaxinc(1:J_r) = hidbiasesMaxinc(index);
hid_visMax_inc(1:J_r,:) = hid_visMax_inc(index,:);        
         
if label == 1
        hid_yMax(1:J_r,:) = hid_yMax(index,:);  
        hid_yMax_inc(1:J_r,:) = hid_yMax_inc(index,:);
end
    



