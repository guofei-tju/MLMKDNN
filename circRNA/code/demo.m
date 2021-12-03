
clear
seed = 12345678;
% rand('seed', seed);
nfolds = 5; nruns=1;
N_features1 = 5;
N_features2 = 5;

%load('dataset.mat')

%% load dataset
dataname = '';
dataname;
% load adjacency matrix
[y,L1,L2] = loadtabfile(['../data/interactions/' dataname 'cd_adjmat.txt']);
%%%%%%%%%%%%%%%%%%%
k1_paths = {['../data/interactions/' dataname 'circRNA_Gene.txt'],...
            ['../data/interactions/' dataname 'circRNA_miRNA.txt'],...
            };
K1 = [];
for i=1:length(k1_paths)
    [y_,l1,l2] = loadtabfile(k1_paths{i});
    K1(:,:,i) = Knormalized(kernel_corr(y_,1,0,1));
end
k2_paths = {['../data/kernels/' dataname 'circSim.txt']
           };
for j=1:length(k2_paths)
    i = i+1;
    [mat, labels] = loadtabfile(k2_paths{j});
    mat = process_kernel(mat);
    K1(:,:,i) = Knormalized(mat);
end


%K1 compelted

k3_paths = {['../data/interactions/' dataname 'disease_Gene.txt'],...

            ['../data/interactions/' dataname 'disease_miRNA.txt'],...
            };
K2 = [];
for i=1:length(k3_paths)
    [y_,l1,l2] = loadtabfile(k3_paths{i});
    K2(:,:,i) = Knormalized(kernel_corr(y_,1,0,1));
end
k4_paths = {['../data/kernels/' dataname 'disease_sim.txt']
           };
for k=1:length(k4_paths)
    i=i+1;
    [mat, labels] = loadtabfile(k4_paths{k});
    mat = process_kernel(mat);
    K2(:,:,i) = Knormalized(mat);
end

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
% nLayers = 1;num_nodes = 1;lambdas = 0.25;sig1s=1;sig2s=1;p2s=4;p3s=1;gammas=32;
% nLayers = 1;num_nodes = 2;lambdas = 0.5;sig1s=1;sig2s=1;p2s=4;p3s=1;gammas=32;
% nLayers = 1;num_nodes = 3;lambdas = 0.5;sig1s=1;sig2s=1;p2s=3;p3s=1;gammas=32;
% nLayers = 1;num_nodes = 4;lambdas = 0.25;sig1s=1;sig2s=1;p2s=3;p3s=1;gammas=32;
% 
% nLayers = 2;num_nodes = 1;lambdas = 0.125;sig1s=1;sig2s=1;p2s=1;p3s=1;gammas=32;
% nLayers = 2;num_nodes = 2;lambdas = 0.5;sig1s=1;sig2s=1;p2s=1;p3s=1;gammas=32;
% nLayers = 2;num_nodes = 3;lambdas = 0.15;sig1s=1;sig2s=1;p2s=5;p3s=1;gammas=32;
nLayers = 2;num_nodes = 4;lambdas = 0.125;sig1s=1;sig2s=1;p2s=4;p3s=1;gammas=8;

for run=1:nruns
    for nLayer = nLayers
        for lambda = lambdas
            for sig1 = sig1s
                for sig2 = sig2s
                    for p2 = p2s  
                        for p3 = p3s
                        for gamma = gammas
                            for maxI =10
                                rand('seed', seed);
                                crossval_idx = crossvalind('Kfold',y(:),nfolds);
                                fold_aupr=[];fold_auc=[];
                                fold_running_time=[];
                                fold_alpha={};fold_beta = {};
                                for fold=1:nfolds
                                    t1 = clock;
                                    train_idx = find(crossval_idx~=fold);
                                    test_idx  = find(crossval_idx==fold);

                                    y_train = y;
                                    y_train(test_idx) = 0;
                                    K1(:,:,4) = Knormalized(getGipKernel(y_train,gamma));
                                    K2(:,:,4) = Knormalized(getGipKernel(y_train',gamma)); 
                                     K1(:,:,5) = Knormalized(kernel_corr(y_train,1,0,1));
                                     K2(:,:,5) = Knormalized(kernel_corr(y_train,2,0,1));
                                     [A_cos_com,err_list,betas1,betas1f,betas2,betas2f] = ...
                                         train(K1,K2,y_train,nLayers,lambda,maxI,...
                                         LR1,LR2,sig1,sig2,p2,p3,N_features1,N_features2,num_nodes);
                                    % [A_cos_com,betas1,betas2] = deepLap_train(K1,K2,y_train,nLayers,lambda1,lambda2,maxI,LR1,LR2,sig);
                                    t2=clock;
                                    fold_running_time = [fold_running_time;etime(t2,t1)];

                                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                                  %% 4. evaluate predictions for AUC and AUPR
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
                                eval([newname_all_predict_results,'=all_predict_results'])
                                mean_aupr = mean(fold_aupr)
                                mean_auc = mean(fold_auc)
                                mean_running_time = mean(fold_running_time)
                                results = cat(1,results,...
                                    [run,num_nodes,maxI,nLayers,lambda,sig1,sig2,p2,p3,gamma,mean_aupr,mean_auc,mean_running_time]);
                                save_results(['circRNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers.txt'],results); 
                                save(['circRNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_alpha'],'fold_alpha'); 
                                save(['circRNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_beta'],'fold_beta');
                                save(['circRNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_predicted_results'],['results_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers']);
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
    results_aupr = mean(results(:,11));
    results_auc = mean(results(:,12));
end