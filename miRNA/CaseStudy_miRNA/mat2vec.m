function [vec] = mat2vec(mat)
%mat2vec 将矩阵向量化操作
vec = reshape(mat,numel(mat),1);
end

