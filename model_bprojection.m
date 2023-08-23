function [OUTPUT] = model_bprojection(INPUT,npca)

% from PCA domain back to x-z domain;

check=size(INPUT);
%disp(check)
% check(2)
if check(1)>1
    INPUT=INPUT';
end

load('PCADATA.mat');
D=coeff(:,1:npca);
OUTPUT=D*(INPUT(1:npca,:))+MED;
%u=size(OUTPUT)
%disp(u)
%OUTPUT=reshape(OUTPUT,check(1),250,125);
