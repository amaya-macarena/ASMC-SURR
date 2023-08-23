function [myPCE] = PCE_update(npca,ypca,data_set)

tic

addpath('/soft/matlab/UQLab_CollabGio')
uqlab;



%nr_data=data_set-200; % size(X)
nr_data=data_set
%nr_val=100; %size Xval;
%nr_fulldata=nr_data+nr_val;

file_session=sprintf('yFD_update_PCE_SESSION_%3.3d_%3.3d.mat',npca,nr_data);
file_design=sprintf('yFD_update_DESIGN_%3.3d_%3.3d.mat',npca,nr_data);
%file_validation=sprintf('yFD_update_VALIDATION_%3.3d_%3.3d.mat',npca,nr_data);

load('X_train_PCE_cum_update.mat');
load('Y_train_PCE_cum_update.mat');


%X(1:nr_data,1:npca)=INPUT(1:nr_data,1:npca);
X(1:nr_data,1:npca)=INPUT(1:nr_data,1:npca);
%Xval(1:nr_val,1:npca)=INPUT(1+nr_data:+nr_data+nr_val,1:npca);

Y(1:nr_data,1:1:ypca)=OUTPUT(1:nr_data,1:ypca); %* 1e9 ;
%Yval(1:nr_val,1:ypca)=OUTPUT(1+nr_data:+nr_data+nr_val,1:1:ypca); %* 1e9 ;

save(file_design,'X','Y');
%save(file_validation,'Xval','Yval');


 for i = 1:npca
     InputOpts.Marginals(i).Name = sprintf('PCA_COMP%d',i);
     InputOpts.Marginals(i).Type = 'KS';
     InputOpts.Marginals(i).Parameters = X(:,i);
 end
myInput = uq_createInput(InputOpts);

%Moments=zeros(npca,2);

%for i =1:npca
%    aux=myInput.Marginals(i).Moments
%    Moments(i,:)=aux
%end

MetaOpts.Type = 'Metamodel';
MetaOpts.MetaType = 'PCE';

%Use experimental design loaded from the data files:

MetaOpts.ExpDesign.X = X;
MetaOpts.ExpDesign.Y = Y;
MetaOpts.Method = 'LARS';
%Set the maximum polynomial degree to 5:
MetaOpts.TruncOptions.MaxInteraction = 2;
MetaOpts.TruncOptions.qNorm = 0.5:.1:0.9;
MetaOpts.Degree = 1:5;
MetaOpts.Display = 3;
%Provide the validation data set to get the validation error:

%MetaOpts.ValidationSet.X = Xval;
%MetaOpts.ValidationSet.Y = Yval;


myPCE = uq_createModel(MetaOpts);

toc

uq_print(myPCE)

%tic
%YPCE1 = uq_evalModel(myPCE,Xval);
%toc

%prior_samp=uq_getSample(N);

%save('prior_init_models.mat','prior_samp')

%save('YPCE_ypca35.mat','YPCE1')

uq_saveSession(file_session)

%    end

%end
% hold off
% 