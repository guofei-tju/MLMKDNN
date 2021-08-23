
clear
seed = 12345678;
nfolds = 5; nruns=1;
%disease_number = [106]
%% load dataset
dataname = '';
dataname;
% load adjacency matrix
[y,L1,L2] = loadtabfile(['../../data/interactions/' dataname 'cd_adjmat.txt']);

%% Parameter setting
l = 1;
LR1 = 1e-2;
LR2 = 1e-2;
globa_true=[];
globa_predict=[];
results = [];
err_list10 = [];
err_list_all = {};

a2 = 1;
a3 = 1;
p2 = 4;
p3 = 1;

maxI = 15;
sig1 = 1;
sig2 = 1;
lambda = 0.125;
gamma = 32;

N_features1 = 5;
N_features2 = 5;
num_nodes = 4;
nLayers = 2;

% %% a RNA rows are set to 0 
% Y = y;
% Y(disease_number,:) = 0;
% y_train = Y;

%% a Disease rows are set to 0 
Y = y';
Y(disease_number,:) = 0;
y_train = Y;

%% kernel construction
k1_paths = {['../../data/interactions/' dataname 'circRNA_Gene.txt'],...
            ['../../data/interactions/' dataname 'circRNA_miRNA.txt'],...
            };
K1 = [];
for i=1:length(k1_paths)
    [y_,l1,l2] = loadtabfile(k1_paths{i});
    K1(:,:,i) = Knormalized(kernel_corr(y_,1,0,1));
end
k2_paths = {['../../data/kernels/' dataname 'circSim.txt']
           };
for j=1:length(k2_paths)
    i = i+1;
    [mat, labels] = loadtabfile(k2_paths{j});
    mat = process_kernel(mat);
    K1(:,:,i) = Knormalized(mat);
end


%K1 compelted

k3_paths = {['../../data/interactions/' dataname 'disease_Gene.txt'],...

            ['../../data/interactions/' dataname 'disease_miRNA.txt'],...
            };
K2 = [];
for i=1:length(k3_paths)
    [y_,l1,l2] = loadtabfile(k3_paths{i});
    K2(:,:,i) = Knormalized(kernel_corr(y_,1,0,1));
end
k4_paths = {['../../data/kernels/' dataname 'disease_sim.txt']
           };
for k=1:length(k4_paths)
    i=i+1;
    [mat, labels] = loadtabfile(k4_paths{k});
    mat = process_kernel(mat);
    K2(:,:,i) = Knormalized(mat);
end

K1(:,:,4) = Knormalized(getGipKernel(y_train,gamma));
K2(:,:,4) = Knormalized(getGipKernel(y_train',gamma)); 
K1(:,:,5) = Knormalized(kernel_corr(y_train,1,0,1));
K2(:,:,5) = Knormalized(kernel_corr(y_train,2,0,1));


%% predict
[A_cos_com,err_list,betas1,betas1f,betas2,betas2f] = ...
    train(K2,K1,y_train,nLayers,lambda,maxI,...
                     LR1,LR2,sig1,sig2,a2,p2,a3,p3,N_features1,N_features2,num_nodes);

                 
%% evaluate
vec_y = mat2vec(y);
vec_y_predict = mat2vec(A_cos_com);

% [X_AUPR,Y_AUPR,tpr,aupr_LGC_A_KA] = perfcurve(vec_y,vec_y_predict,1, 'xCrit', 'reca', 'yCrit', 'prec');
% [X_AUC,Y_AUC,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(vec_y,vec_y_predict,1);
[X_AUPR,Y_AUPR,tpr,aupr_LGC_A_KA] = perfcurve(vec_y,vec_y_predict,1, 'xCrit', 'reca', 'yCrit', 'prec');
[X_AUC,Y_AUC,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(vec_y,vec_y_predict,1);
A_cos_com_T = A_cos_com';

fprintf('the overall aupr is: %f \n',aupr_LGC_A_KA);
fprintf('the overall auc is: %f \n',AUC_LGC_KA);