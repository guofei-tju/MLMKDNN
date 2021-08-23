
clear
seed = 12345678;
nfolds = 5; nruns=1;
RNA_number = 65;
%% load dataset
load('dataset.mat')
y =  miRNA_disease_Y;  
%% Parameter setting
l = 1;
LR1 = 5e-3;
LR2 = 5e-3;
globa_true=[];
globa_predict=[];
results = [];
err_list10 = [];
err_list_all = {};

a2 = 1;
a3 = 1;
p2 = 1;
p3 = 1;

maxI = 20;
sig1 = 1;
sig2 = 1;
lambda = 0.125;
gamma = 32;

N_features1 = 3;
N_features2 = 3;
num_nodes = 2;
nLayers = 2;

%% a RNA rows are set to 0
Y = y;
Y(RNA_number,:) = 0;
y_train = Y;

%% kernel construction
K1 = [];
K1(:,:,1) = Knormalized(process_kernel(miRNA_Function_S));
K1(:,:,2) = Knormalized(process_kernel(miRNA_Sequences_Needle_S));

K2 = [];
K2(:,:,1) = Knormalized(process_kernel(disease_Function_S));
K2(:,:,2) = Knormalized(process_kernel(disease_Sem_S));
K1(:,:,3) = Knormalized(getGipKernel(y_train,gamma));
K2(:,:,3) = Knormalized(getGipKernel(y_train',gamma));


%% predict
[A_cos_com,err_list,betas1,betas1f,betas2,betas2f] = ...
    train(K1,K2,y_train,nLayers,lambda,maxI,...
                     LR1,LR2,sig1,sig2,a2,p2,a3,p3,N_features1,N_features2,num_nodes);

                 
%% evaluate
vec_y = mat2vec(y);
vec_y_predict = mat2vec(A_cos_com);

[X_AUPR,Y_AUPR,tpr,aupr_LGC_A_KA] = perfcurve(vec_y,vec_y_predict,1, 'xCrit', 'reca', 'yCrit', 'prec');
[X_AUC,Y_AUC,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(vec_y,vec_y_predict,1);
A_cos_com_T = A_cos_com';

fprintf('the overall aupr is: %f \n',aupr_LGC_A_KA);
fprintf('the overall auc is: %f \n',AUC_LGC_KA);