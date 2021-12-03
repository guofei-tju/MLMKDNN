
clear
seed = 12345678;
% rand('seed', seed);
nfolds = 5; nruns=1;
N_features1 = 3;
N_features2 = 3;

%load('dataset.mat')

%% load dataset
%%%%%%%%%%%%%%%%%%%
load('../data/disease_sim_2017.mat');
load('../data/lncR_disease_2017.mat');
load('../data/lncR_sim_2017.mat');
load('../data/lncR_SequenceSim_2017.mat')
lncSim = lncR_sim_matrix;
lncSeqSim = lncR_Seqsim_2017;
disSim_Jaccard = disease_sim_matrix;
interMatrix = lncR_disease_matrix;
y = interMatrix;

K1 = [];
mat = process_kernel(lncSim);
K1(:,:,1)=Knormalized(mat);
% mat = process_kernel(lncSeqSim);
% K1(:,:,4)=Knormalized(mat);
K2 = [];
mat = process_kernel(disSim_Jaccard);
K2(:,:,1)=Knormalized(mat);

%%               
LR1 = 1e-2;
LR2 = 1e-2;
sig = 1;

globa_true=[];
globa_predict=[];
results = [];
err_list10 = [];
err_list_all = {};
l = 1;
a2 = 1;
a3 = 1;
sig1 = 1;
sig2 = 1;

% nLayers = 1;num_nodes = 1;lambdas = 1;sig1s=1;sig2s=1;a2s=1;p2s=1;a3s =1 ;p3s=1;gammas=32;
% nLayers = 1;num_nodes = 2;lambdas = 0.25;sig1s=1;sig2s=1;a2s=1;p2s=1;a3s =1 ;p3s=1;gammas=32;
% nLayers = 1;num_nodes = 3;lambdas = 0.25;sig1s=1;sig2s=1;a2s=1;p2s=1;a3s =5 ;p3s=1;gammas=32;
% nLayers = 1;num_nodes = 4;lambdas = 0.25;sig1s=1,sig2s = 1;a2s=1;p2s=5;a3s =5 ;p3s=1;gammas=32;
% 
% nLayers = 2;num_nodes = 1;lambdas = 0.125;sig1s=1;sig2s=1;a2s=1;p2s=1;a3s =1 ;p3s=1;gammas=32;
% nLayers = 2;num_nodes = 2;lambdas = 0.125;sig1s=1;sig2s=1;a2s=1;p2s=1;a3s =1 ; p3s=1;gammas=32;
% nLayers = 2;num_nodes = 3;lambdas = 0.125;sig1s=1;sig2s =1;a2s=1;p2s=1;a3s =1;p3s=1;gammas=32;
nLayers = 2;num_nodes = 4;lambdas = 0.125;sig1s=1;sig2s=1;a2s =1; p2s=1;a3s =1;p3s=1;gammas=32;

for run=1:nruns
    for nLayer = nLayers
        for lambda = lambdas
            for sig1 = sig1s
                for sig2 = sig2s
                    for a2 = a2s
                    for p2 = p2s 
                        for a3= a3s
                        for p3 = p3s
                            for gamma = gammas
                                for maxI = 10
                                    rand('seed', seed);
                                    crossval_idx = crossvalind('Kfold',y(:),nfolds);
                                    fold_aupr=[];fold_auc=[];fold_running_time=[];
                                    fold_alpha={};fold_beta = {};
                                    for fold=1:nfolds
                                        t1 = clock;
                                        train_idx = find(crossval_idx~=fold);
                                        test_idx  = find(crossval_idx==fold);

                                        y_train = y;
                                        y_train(test_idx) = 0;

                                        K1(:,:,2) = Knormalized(getGipKernel(y_train,gamma));
                                        K2(:,:,2) = Knormalized(getGipKernel(y_train',gamma));

                                        [KD, KL] = consine(y_train');
                                        K1(:,:,3)= Knormalized(KL);
                                        K2(:,:,3)= Knormalized(KD);

                                         [A_cos_com,err_list,betas1,betas1f,betas2,betas2f] = ...
                                             train(K1,K2,y_train,nLayers,lambda,maxI,...
                                             LR1,LR2,sig1,sig1,a2,p2,a3,p3,N_features1,N_features2,num_nodes);
                                        % [A_cos_com,betas1,betas2] = deepLap_train(K1,K2,y_train,nLayers,lambda1,lambda2,maxI,LR1,LR2,sig);
                                        t2=clock;
                                        fold_running_time = [fold_running_time;etime(t2,t1)];

                                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                                      %% 4. evaluate predictions
                                        yy=y;
                                        %yy(yy==0)=-1;
                                        %stats = evaluate_performance(y2(test_idx),yy(test_idx),'classification');
                                        test_labels = yy(test_idx);
                                        predict_scores = A_cos_com(test_idx);
                                        [X,Y,tpr,aupr_LGC_A_KA] = perfcurve(test_labels,predict_scores,1, 'xCrit', 'reca', 'yCrit', 'prec');

                                        [X,Y,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_labels,predict_scores,1);

                                        fprintf('---------------\nRUN %d - FOLD %d  \n', run, fold)

                                        fprintf('%d - FOLD %d - AUPR: %f - AUC: %f\n', run, fold, aupr_LGC_A_KA,AUC_LGC_KA )
                                        % For alpha and beta 
                                        fold_alpha = [fold_alpha,{betas1;betas1f}];
                                        fold_beta = [fold_beta,{betas2;betas2f}];

                                        fold_aupr=[fold_aupr;aupr_LGC_A_KA];
                                        fold_auc=[fold_auc;AUC_LGC_KA];
                                        err_list10 = [err_list10;err_list];

                                        globa_true=[globa_true;test_labels];
                                        globa_predict=[globa_predict;predict_scores];
                                    end
                                    all_predict_results = [globa_true,globa_predict];
                                    newname_all_predict_results = ['results_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers'];
                                    eval([newname_all_predict_results,'=all_predict_results']);


                                    mean_aupr = mean(fold_aupr)
                                    mean_auc = mean(fold_auc)
                                    mean_running_time = mean(fold_running_time)
                                    results = cat(1,results,...
                                        [run,num_nodes,maxI,nLayers,lambda,sig1,sig2,a2,p2,a3,p3,gamma,mean_aupr,mean_auc]);
                                    save_results(['lnc2RNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_poly2.txt'],results); 
%                                     save(['lnc2RNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_alpha'],'fold_alpha'); 
%                                     save(['lnc2RNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_beta'],'fold_beta');
                                    save(['lnc2RNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_predicted_results'],['results_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers']);

                                    err_list_all{l} = err_list10;
                                    l = l+1;
                                    err_list10 = [];
                                end          
                            end
                        end
                        end
                    end
                    end
                end
            end            
        end
    end
end