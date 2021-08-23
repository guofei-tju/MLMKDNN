function [K,Kf] = MN_computeKernels(K_list,betas,num_nodes,nLayers,sig,p2,p3)

dotx = K_list;
r = size(K_list,1);
K = zeros(r*r,num_nodes,nLayers);
Kf = zeros(r*r,nLayers);

for t=1:nLayers

    Krbf = normalizeKernel(rbf2(dotx,sig));
    Kpoly2 = normalizeKernel((dotx+p2).^2);
    Kpoly3 = normalizeKernel((dotx+p3).^3);
    Klin = normalizeKernel(dotx);

    switch num_nodes
        case 1
             K(:,1,t) = Kpoly2(:);
        case 2
             K(:,1,t) = Kpoly2(:);
             K(:,2,t) = Klin(:);
        case 3
            K(:,1,t) = Kpoly2(:);
            K(:,3,t) = Kpoly3(:);
            K(:,2,t) = Klin(:);
        case 4
            K(:,4,t) = Krbf(:);
            K(:,1,t) = Kpoly2(:);
            K(:,3,t) = Kpoly3(:);
            K(:,2,t) = Klin(:);
    end
    dotx = K(:,:,t).*repmat(betas(t,:),r*r,1);
    Kf(:,t) = sum(dotx,2);
    dotx = normalizeKernel(reshape(Kf(:,t),r,r));
end

