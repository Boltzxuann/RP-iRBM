clear
close all
load BinaryDataMNIST
load best_vh
data = Bdata_test;
numcases = size(data,1);
beta = beta0 * soft_plus(0*hb);
P_z_on_v = P_z( data , hv ,hb , numh , beta ,beta0, numcases ); 
[~,Max_z] = max(P_z_on_v);
zz = unique(Max_z);
histogram_z = histc(Max_z,zz);




