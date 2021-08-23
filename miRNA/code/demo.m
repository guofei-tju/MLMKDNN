
clear
seed = 12345678;
% rand('seed', seed);
nfolds = 5; nruns=1;
N_features1 = 3;
N_features2 = 3;
num_nodes = 2;
load('../data/dataset.mat')


K1 = [];
K1(:,:,1) = Knormalized(process_kernel(miRNA_Function_S));
K1(:,:,2) = Knormalized(process_kernel(miRNA_Sequences_Needle_S));

K2 = [];
K2(:,:,1) = Knormalized(process_kernel(disease_Function_S));
K2(:,:,2) = Knormalized(process_kernel(disease_Sem_S));
y =  miRNA_disease_Y;  
              
LR1 = 1e-2;
LR2 = 1e-2;
sig = 1;

globa_true=[];
globa_predict=[];
results = [];
err_list10 = [];
err_list_all = {};
l = 1;
for run=1:nruns
    for nLayers = 1
        for lambda = 0.125%[2^-3,2^-2,2^-1,1,2,2^2,2^3]%[0.5 0.75 1 1.25 1.5]%[2^-5,2^-4,2^-3,2^-2,2^-1,1,2,2^2,2^3,2^4,2^5]
            for sig1 = 1%[1,2,2^2,2^3,2^4]%[2^-3,2^-2,2^-1,1,2,2^2,2^3]%[0.5 0.75 1 1.25 1.5]%[2^-5,2^-4,2^-3,2^-2,2^-1,1,2,2^2,2^3,2^4,2^5]
                for sig2 = 1%[1,2,2^2,2^3,2^4]
                    for p2 = 5%[1,2,3,4,5]
                        for p3 = 1%[1,2,3,4,5]
                        for gamma = 32%[2^-3,2^-2,2^-1,1,2,2^2,2^3]
                              for maxI = 10
                                rand('seed', seed);
                                crossval_idx = crossvalind('Kfold',y(:),nfolds);
                                fold_aupr=[];fold_auc=[];
                                for fold=1:nfolds
                                    t1 = clock;
                                    train_idx = find(crossval_idx~=fold);
                                    test_idx  = find(crossval_idx==fold);

                                    y_train = y;
                                    y_train(test_idx) = 0;
                                    K1(:,:,3) = Knormalized(getGipKernel(y_train,gamma));
                                    K2(:,:,3) = Knormalized(getGipKernel(y_train',gamma));
%                                     [KD, KL] = GaussianKernel(y_train', 1, 1);
%                                     K1(:,:,3)=Knormalized(KL);
%                                     K2(:,:,3)=Knormalized(KD);
                                     [A_cos_com,err_list,betas1,betas1f,betas2,betas2f] = ...
                                         train(K1,K2,y_train,nLayers,lambda,maxI,...
                                         LR1,LR2,sig1,sig2,p2,p3,N_features1,N_features2,num_nodes);
                                    % [A_cos_com,betas1,betas2] = deepLap_train(K1,K2,y_train,nLayers,lambda1,lambda2,maxI,LR1,LR2,sig);
                                    t2=clock;
                                    etime(t2,t1)

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

                                    fold_aupr=[fold_aupr;aupr_LGC_A_KA];
                                    fold_auc=[fold_auc;AUC_LGC_KA];
                                    err_list10 = [err_list10;err_list];

                                    globa_true=[globa_true;test_labels];
                                    globa_predict=[globa_predict;predict_scores];
                                end
                                mean_aupr = mean(fold_aupr)
                                mean_auc = mean(fold_auc)

                                results = cat(1,results,...
                                    [run,num_nodes,maxI,nLayers,lambda,sig1,sig2,p2,p3,gamma,mean_aupr,mean_auc]);
                                save_results(['miRNA_disease_',num2str(num_nodes),'_nodes_',num2str(nLayers),'_Layers_.txt'],results); 

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