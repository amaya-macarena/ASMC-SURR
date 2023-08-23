function [myPCE,myInput,Moments] = PCE_training(npca,ypca,data_set,N)
addpath('/soft/matlab/UQLab_CollabGio')
uqlab;

tic
uqlab('yFD_69tt_PCE_SESSION_Loo_100_200.mat')
toc

myPCE = uq_getModel
myInput= uq_getInput

Moments=zeros(npca,2);

for i =1:npca
    aux=myInput.Marginals(i).Moments
    Moments(i,:)=aux
end

prior_samp=uq_getSample(N);

save('prior_init_models.mat','prior_samp')