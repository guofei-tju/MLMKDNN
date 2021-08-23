function [betas,betasf] = grad_1_Layer(betas,betasf,KCOMList1,KNet1,...
                                       KCOMList1_all,KCOMList1f,...
                                       K2_final,Theta,Ytrain,LR,...
                                       num_nodes,N_features,flag)
  
    if flag ==0
        r = size(Ytrain,1);
    elseif flag ==1
        r = size(Ytrain,2);     
    end
    
    Dbetas = zeros(1,num_nodes,N_features);
    Dbetas3 = zeros(1,N_features);
    
    Kd = zeros(r,r,num_nodes,1,N_features);
    
    for i = 1:N_features
        K1 = KCOMList1_all(:,i);
        Dbetas3(i) = MNMLMKL_Deriv(K1,KCOMList1f,K2_final,Theta,Ytrain,flag);       
    end 
    
    for i = 1:N_features
        for k = 1:num_nodes  
            Dbetas(1,k,i) = MNMLMKL_Deriv(KNet1(:,k,1,i),KCOMList1f,K2_final,Theta,Ytrain,flag);         
        end
    end
    betas = betas - LR * Dbetas;
    betasf = betasf - LR * Dbetas3 ;
    %归一化
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