
hidvis1 = hid_visMax(1:numhid,:);
norm_hidvis = sqrt(  sum (hidvis1.^2, 2)  ) ;
mean_norm = mean( norm_hidvis );

hids_onoff = norm_hidvis > 0.1* mean_norm;
index= find(hids_onoff);

vishid = hidvis1 (index,:)';
if label == 1
         labhid = hid_yMax(index,:)';
         labbiases = ybiases;
end
hidbiases = hidbiasesMax(index);
