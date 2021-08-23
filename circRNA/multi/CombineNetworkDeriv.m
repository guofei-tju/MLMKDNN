function answ = CombineNetworkDeriv(Kf,betas)

    [r,~] = size(Kf); r=sqrt(r);
    Kf = reshape(Kf,r,r);
    answ = betas * Kf;
    
end

