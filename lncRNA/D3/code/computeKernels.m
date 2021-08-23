function [K,Kf] = computeKernels(K_list,betas,nLayers)

r = size(K_list,1);
K = zeros(r*r,4,nLayers);
Kf = zeros(r*r,nLayers);

for t=1:size(K_list,3)
    KK = K_list(:,:,t);
    K(:,t,1) = KK(:);
end
dotx = K(:,:,1).*repmat(betas(1,:),r*r,1);
Kf(:,1) = sum(dotx,2);
dotx = reshape(Kf(:,1),r,r);
for t=2:nLayers

    Krbf = normalizeKernel(rbf2(dotx,0.5));
    Kpoly2 = normalizeKernel((dotx+1).^2);
    Kpoly3 = normalizeKernel((dotx+1).^3);
    Klin = normalizeKernel(dotx);
    
    K(:,1,t) = Krbf(:);
    K(:,2,t) = Kpoly2(:);
    K(:,3,t) = Kpoly3(:);
    K(:,4,t) = Klin(:);

    dotx = K(:,:,t).*repmat(betas(t,:),r*r,1);
    Kf(:,t) = sum(dotx,2);
    dotx = reshape(Kf(:,t),r,r);
end

