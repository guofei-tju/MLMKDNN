
clear
seed = 12345678;
nfolds = 5; nruns=1;
RNA_name = 65;
%% load dataset
%%%%%%%%%%%%%%%%%%%
lncRNAsimilarity  = importdata('../../data/lncRNAsimilarity.txt');
diseasesimilarity=importdata('../../data/diseasesimilarity.txt');
geneDis = load('../../data/known_gene_disease_interaction.txt');
interMatrix = load('../../data/known_lncRNA_disease_interaction.txt');
lncSim = lncRNAsimilarity;
disSim_Jaccard = diseasesimilarity;
y = interMatrix;

K1 = [];
mat = process_kernel(lncSim); 
K1(:,:,1)=Knormalized(mat);
K1(:,:,2) = Knormalized(process_kernel(ncRNASS(interMatrix, disSim_Jaccard)));
K2 = [];
mat = process_kernel(disSim_Jaccard);
K2(:,:,1)=Knormalized(mat);
K2(:,:,2)= Knormalized(process_kernel(kernel_corr(geneDis,2,0,1)));


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
p2 = 5;
p3 = 1;

maxI = 20;
sig1 = 1;
sig2 = 1;
lambda = 0.125;
gamma = 32;

N_features1 = 3;
N_features2 = 3;
num_nodes = 3;
nLayers = 2;

%% a RNA rows are set to 0
Y = y;
Y(RNA_name,:) = 0;
y_train = Y;

%% kernel construction

K1(:,:,3)=Knormalized(kernel_corr(y_train,1,0,1));
K2(:,:,3)=Knormalized(kernel_corr(y_train,2,0,1));

%% predict
[A_cos_com,err_list,betas1,betas1f,betas2,betas2f] = ...
    train(K1,K2,y_train,nLayers,lambda,maxI,...
                     LR1,LR2,sig1,sig2,a2,p2,a3,p3,N_features1,N_features2,num_nodes);

save  A_cos_com A_cos_com           
%% evaluate
vec_y = mat2vec(y);
vec_y_predict = mat2vec(A_cos_com);

[X_AUPR,Y_AUPR,tpr,aupr_LGC_A_KA] = perfcurve(vec_y,vec_y_predict,1, 'xCrit', 'reca', 'yCrit', 'prec');
[X_AUC,Y_AUC,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(vec_y,vec_y_predict,1);
A_cos_com_T = A_cos_com';

fprintf('the overall aupr is: %f \n',aupr_LGC_A_KA);
fprintf('the overall auc is: %f \n',AUC_LGC_KA);