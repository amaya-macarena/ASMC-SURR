function [YPCE,prior_density] = PCE_eval(X,myPCE,myInput)
addpath('/soft/matlab/UQLab_CollabGio')
%calculate prior probability and evaluate the surrogate
%tic
%load('PCE2_element.mat','myPCE')
%toc
%X=zeros(35);
%addpath(genpath('C:\Users\mamaya\Documents\Third_project\ASMC-SGR-main\call_matlab'))
%uqlab
%tic
%uqlab('yFD_PCE_SESSION_050_900.mat');
%toc

prior_density=uq_evalPDF(X, myInput);

prior_density=log(prior_density);

%tic
YPCE = uq_evalModel(myPCE,X);
%toc