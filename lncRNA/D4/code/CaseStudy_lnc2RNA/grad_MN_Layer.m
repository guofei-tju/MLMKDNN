function [betas,betasf] = grad_MN_Layer(betas,betasf,KCOMList1,KNet1,...
                                         KCOMList1_all,K1COM,K2_final,...
                                         Theta,Ytrain,LR,sig,p2,p3,...
                                         num_nodes,N_features,flag)
  
    if flag ==0
        r = size(Ytrain,1);
    elseif flag ==1
        r = size(Ytrain,2);     
    end
    
    Dbetas = zeros(2,num_nodes,N_features);
    Dbetas3 = zeros(1,N_features);
    Kd = zeros(r,r,num_nodes,2,N_features);
    for i = 1:N_features
        K1 = KCOMList1_all(:,i);
        Dbetas3(i) = MNMLMKL_Deriv(K1,K1COM,K2_final,Theta,Ytrain,flag);       
    end    
    %N_features个网络，每个网络两层，每层四个节�?
    for i = 1:N_features
        for l = 2:-1:1
            for k = 1:num_nodes
                switch l
                    case 1
                        Kd(:,:,k,l,i) = TwoLayerDeriv(KCOMList1(:,1,i),KNet1(:,k,1,i),betasf(i),betas(2,:,i),sig,p2,p3,num_nodes); 
                    case 2
                        Kd(:,:,k,l,i) = CombineNetworkDeriv(KNet1(:,k,2,i),betasf(i));
                end           
                Dbetas(l,k,i) = MNMLMKL_Deriv(Kd(:,:,k,l,i),K1COM,K2_final,Theta,Ytrain,flag);           
            end
        end
    end
    betas = betas - LR * Dbetas;
    betasf = betasf - LR * Dbetas3 ;      
    %归一�?
    for i = 1:size(betas,3)
        for j = 1:size(betas,1)
            betas_1 = betas(j,:,i);
            betas_1(betas_1<0)=1e-10; %non-negative
            if sum(betas_1(:))~=1
                betas(j,:,i) = betas_1/sum(betas_1(:)); %trace final layer upper bound
            end
        end
    end
    betasf = betasf/sum(betasf);

end

function [dLdu] = MNMLMKL_Deriv(K1,K1COM,K2COM,Theta,Ytrain,flag)
    if flag == 0
        r = size(Ytrain,1);
        K1 = reshape(K1,r,r);
        K1COM = reshape(K1COM,r,r);
        dLdu = -2*trace(Ytrain'*K1*Theta*K2COM)+trace(K2COM'*Theta'*K1*K1COM*Theta*K2COM)+trace(K2COM'*Theta'*K1COM'*K1*Theta*K2COM);
    elseif flag == 1
        r = size(Ytrain,2);
        K1 = reshape(K1,r,r);
        K1COM = reshape(K1COM,r,r);

        dLdu = -2*trace(Ytrain'*K2COM*Theta*K1)+trace(K1'*Theta'*K2COM'*K2COM*Theta*K1COM)+trace(K1COM'*Theta'*K2COM'*K2COM*Theta*K1);           
    end
end