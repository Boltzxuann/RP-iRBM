
hidbiases = hbMax(1:numhid);
vishid = h_vMax(1:numhid,:).'; y_hid = h_yMax(1:numhid,:).'; 
numclasses =10;
data_test= Bdata_train(50001:60000,:);

p_y_on_x_tst=zeros(10000,10);
for t=1:10000
   midterm_y=hidbiases+data_test(t,:)*vishid ;

   p_m = sum( log( 1+exp(repmat(midterm_y,numclasses,1)+y_hid)' ) ) +yb ;
   max_p_m = max (  p_m );
   p_m_maxpm = p_m - repmat( max_p_m ,1 , numclasses );
   ln_partition = max_p_m + log( sum(exp(p_m_maxpm)) );
   %partition=sum(exp(sum(log(1+exp(repmat(midterm_y,10,1)+M_y_hid)'))+M_ybiases));%%·ÖÄ¸
   p_y_on_x_tst(t,:) =  exp ( p_m - repmat(ln_partition,1, numclasses ) );   
   %p_y_on_x_tst(t,c)=exp(M_ybiases(1,c)+sum(log(1+exp(M_y_hid(c,:)+midterm_y))))/partition;

end
[P_max,target]=max(p_y_on_x_tst');
[one,TstLbls]=max(train_targets(50001:60000,:)');
correct=target==TstLbls;
TestAccuracy=sum(correct)/10000;