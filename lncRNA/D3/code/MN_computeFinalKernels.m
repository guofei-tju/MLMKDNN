function [Kf] = MN_computeFinalKernels(K_list,betas)

r2 = size(K_list,1);r = sqrt(r2);
dotx = K_list.*repmat(betas,r2,1);
Kf = sum(dotx,2);

end
