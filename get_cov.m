function [COV_PCE_TOT] = get_cov(npca,data_set,myPCE,update_surr)

addpath('/soft/matlab/UQLab_CollabGio')
%uqlab('yFD_update_PCE_SESSION_050_900.mat')
%uqlab('yFD_PCE_SESSION_050_900.mat')

load('X_train_PCE_cum_update.mat')

load('Y_train_PCE_cum_update.mat')

%myPCE = uq_getModel

%down=data_set-200+1  %100 models

YY=[myPCE.Error.LooPred]

down=1
up=data_set

%YPCE = uq_evalModel(myPCE,INPUT(down:up,1:npca));

%data_set+100 is for validation, then data_set+200 for covariance

%A=YPCE(:,:)-OUTPUT(down:up,:)%*1e9;

A=YY-OUTPUT(down:up,:)%*1e9;

MEAN_Y=mean(A,1);

COV_PCE_MOD=cov(YY-MEAN_Y-OUTPUT(down:up,:))%*1e9);

COV_PCE_TOT=COV_PCE_MOD+eye(69)*(0.5^2)%*1e-18;

%COV_MOD=COV_PCE;

COV=sprintf('UPDATED_COV%d.mat',update_surr);
save(COV,'COV_PCE_TOT')%,'COV_PCE_MOD','MEAN_Y');