function [Y,err_list,betas1,betas1f,betas2,betas2f] = train(K1_list,K2_list,...
          y_train,nLayers,lambda,maxI,LR1,LR2,sig1,sig2,a2,p2,a3,p3,N_features1,N_features2,num_nodes)

%%%%%%%%%%%%%%%%%%%%%%%%%
%   intruduction:
%   each target and drug generate four network using features respectivelyï¼?
%   and combine the outputs of the networks for each target and  drug as
%   the final output.
%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1 initialize weights
% average initialize
% num_nodes = 4;
betas1 = zeros(nLayers,num_nodes,N_features1);
betas2 = zeros(nLayers,num_nodes,N_features2);

for i = 1:N_features1
    betas1(1:nLayers,:,i) = ones(nLayers,num_nodes)./num_nodes;
%     betas1(:,:,i) = rand_initialize(nLayers,4);   
end
for i = 1:N_features2
    betas2(1:nLayers,:,i) = ones(nLayers,num_nodes)./num_nodes;
%     betas2(:,:,i) = rand_initialize(nLayers,4);
end
% betas1f = rand_initialize(1,N_features1);
% betas2f = rand_initialize(1,N_features1);

betas1f = ones(1,N_features1)./N_features1;
betas2f = ones(1,N_features2)./N_features2;

%alternating opt
err_list = [];
[r,c] = size(y_train);
%% 2 forward calculation
%lncRNA side
for i = 1:N_features1   
    [KNet1(:,:,:,i),KCOMList1(:,:,i)] = MN_computeKernels(K1_list(:,:,i),betas1(:,:,i),num_nodes,nLayers,sig1,a2,p2,a3,p3);
    KCOMList1_all(:,i) = KCOMList1(:,nLayers,i);
end
[KCOMList1f] = MN_computeFinalKernels(KCOMList1_all,betas1f);
K1_final = reshape(KCOMList1f(:),r,r);

% disease side 
for i = 1:N_features2
    [KNet2(:,:,:,i),KCOMList2(:,:,i)] = MN_computeKernels(K2_list(:,:,i),betas2(:,:,i),num_nodes,nLayers,sig1,a2,p2,a3,p3);
    KCOMList2_all(:,i) = KCOMList2(:,nLayers,i);
end
[KCOMList2f] = MN_computeFinalKernels(KCOMList2_all,betas2f);
K2_final = reshape(KCOMList2f(:),c,c); 

% y_train = preprocess_WKNKN(y_train,K1_final,K2_final,1,0.5);
%% 3 graident desent
for i = 1:maxI

    %% calculate THETA
    A = K1_final*K1_final';
    B = lambda*pinv(K2_final'*K2_final);
    C = -K1_final'*y_train * pinv(K2_final');
    Theta = lyap(A,B,C);
    
    %% calculate J
    J = norm(y_train-K1_final*Theta*K2_final','fro')^2+lambda*norm(Theta,'fro')^2
    err_list = [err_list,J'];

    %% calculate delta_beta
    if nLayers==1
         [betas1,betas1f] = grad_1_Layer(betas1,betas1f,KCOMList1,KNet1,...
                                          KCOMList1_all,KCOMList1f,...
                                          K2_final,Theta,y_train,LR1,...
                                          num_nodes,N_features1,0);
                           
         [betas2,betas2f] = grad_1_Layer(betas2,betas2f,KCOMList2,KNet2,...
                                          KCOMList2_all,KCOMList2f,...
                                          K1_final,Theta,y_train,LR2,...
                                          num_nodes,N_features2,1);                                             
    end
    if nLayers==2
         [betas1,betas1f] = grad_MN_Layer(betas1,betas1f,KCOMList1,KNet1,...
                                          KCOMList1_all,KCOMList1f,K2_final,...
                                          Theta,y_train,LR1,sig1,p2,p3,...
                                          num_nodes,N_features1,0);
                           
         [betas2,betas2f] = grad_MN_Layer(betas2,betas2f,KCOMList2,KNet2,...
                                          KCOMList2_all,KCOMList2f,K1_final,...
                                          Theta,y_train,LR2,sig2,p2,p3,...
                                          num_nodes,N_features2,1);                                             
    end
    
%% forward calculation
    %lncRNA side
    for i = 1:N_features1   
        [KNet1(:,:,:,i),KCOMList1(:,:,i)] = MN_computeKernels(K1_list(:,:,i),betas1(:,:,i),num_nodes,nLayers,sig1,a2,p2,a3,p3);
        KCOMList1_all(:,i) = KCOMList1(:,nLayers,i);
    end
    [KCOMList1f] = MN_computeFinalKernels(KCOMList1_all,betas1f);
    K1_final = reshape(KCOMList1f(:),r,r);

    % disease side 
    for i = 1:N_features2
        [KNet2(:,:,:,i),KCOMList2(:,:,i)] = MN_computeKernels(K2_list(:,:,i),betas2(:,:,i),num_nodes,nLayers,sig1,a2,p2,a3,p3);
        KCOMList2_all(:,i) = KCOMList2(:,nLayers,i);
    end
    [KCOMList2f] = MN_computeFinalKernels(KCOMList2_all,betas2f);
    K2_final = reshape(KCOMList2f(:),c,c);  

end
%% calculate Y
Y = K1_final*Theta*K2_final';
end

