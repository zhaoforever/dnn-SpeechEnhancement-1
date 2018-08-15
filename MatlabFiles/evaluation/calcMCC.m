function mcc = calcMCC(specRef,specTest,dim)
%calcMCC   Calculate the spectral magnitude normalization coefficient

if nargin < 3 || isempty(dim); dim = 1; end

xx    = bsxfun(@minus,specRef, mean(specRef, dim));
denom = sqrt(sum(xx.^2,dim));
denom = denom + (denom == 0);
xx    = bsxfun(@rdivide,xx,denom);

yy    = bsxfun(@minus,specTest,mean(specTest,dim));
denom = sqrt(sum(yy.^2,dim));
denom = denom + (denom == 0);
yy    = bsxfun(@rdivide,yy,denom);

mcc   = sum(xx .* yy, dim);
%mcc   = xx .* yy;