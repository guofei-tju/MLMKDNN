function answ = TwoLayerDeriv(Kf,K,betas,betas2,sig,p2,p3,num_nodes)

    [r,~] = size(Kf); r=sqrt(r);
    Kf = reshape(Kf,r,r); K = reshape(K,r,r);
    Kf = betas * Kf;
    switch num_nodes
        case 1
             answ = betas2(1).*normalizeKernel_Grad(Kf,Two_polyDeriv(Kf,K,p2));
        case 2
            answ = betas2(1)*normalizeKernel_Grad(Kf,Two_polyDeriv(Kf,K,p2))...
                    + betas2(2)*normalizeKernel_Grad(Kf,Two_linDeriv(Kf));
        case 3
            answ = betas2(1)*normalizeKernel_Grad(Kf,Two_polyDeriv(Kf,K,p2))...
                + betas2(3)*normalizeKernel_Grad(Kf,Two_polyDeriv2(Kf,K,p3))...
                + betas2(2)*normalizeKernel_Grad(Kf,Two_linDeriv(Kf));
        case 4
            answ = betas2(4).*normalizeKernel_Grad(Kf,Two_rbfDeriv(Kf,K,sig))...
                + betas2(1)*normalizeKernel_Grad(Kf,Two_polyDeriv(Kf,K,p2))...
                + betas2(3)*normalizeKernel_Grad(Kf,Two_polyDeriv2(Kf,K,p3))...
                + betas2(2)*normalizeKernel_Grad(Kf,Two_linDeriv(Kf));
    end
end

function answ = Two_polyDeriv(Kf,K,p2)
    answ = 2.*(Kf+p2).*K;
end

function answ = Two_polyDeriv2(Kf,K,p3)
    answ = 3.*((Kf+p3).^2).*K;
end

function answ = Two_rbfDeriv(Kf,K,sig)
    answ = exp(-2/(2*sig^2).*(1-Kf)).*(2/(2*sig^2).*K);
end

function answ = Two_linDeriv(Kf)
    answ = Kf;
end

function answ = normalizeKernel_Grad(Kf,KfDeriv)
dKf = diag(Kf);
dKfDeriv = diag(KfDeriv);
answ = KfDeriv.*((dKf*dKf').^0.5)...
    +Kf.*(-0.5.*(dKf*dKf').^(1.5))...
    .*(dKfDeriv*dKf')+(dKf*dKfDeriv');
end
