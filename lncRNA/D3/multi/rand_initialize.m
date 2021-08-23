function [beta] = rand_initialize(nLayers,N_features1)

beta = rand(nLayers,N_features1);

for i = 1:nLayers
    beta(i,:) = beta(i,:)/sum(beta(i,:));
end

end

