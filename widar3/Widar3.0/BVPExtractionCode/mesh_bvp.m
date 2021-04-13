clear;
clc;
close all;

VS = cell2mat(struct2cell(load('BVP/user-user-1-1-1-1-1-1e-07-100-20-100000-L0.mat')));
for ii = 1:size(VS,3)
    mesh(squeeze(VS(:,:,ii)));view([30,45]);
    % Breakpoint here
end