function [ ] = im_gray(corn_gray)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
meanIntensity = mean(corn_gray(:));
corn_binary = corn_gray > meanIntensity;
%imshow(corn_binary)
imshow(imresize(corn_binary,[50,nan]))
end

