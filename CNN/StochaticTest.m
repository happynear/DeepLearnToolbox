function pooledFeatures = StochaticTest(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);
    
maxPosition = zeros(convolvedDim,convolvedDim,numFilters,numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%

sumFilter = ones(poolDim,poolDim);
for i=1:numFilters
    for j=1:numImages
          sumSquareImage=conv2(squeeze(convolvedFeatures(:,:,i,j).^2),sumFilter,'valid');
          sumImage=conv2(squeeze(convolvedFeatures(:,:,i,j)),sumFilter,'valid');
          pooledFeatures(:,:,i,j)=sumSquareImage(1:poolDim:end,1:poolDim:end)./sumImage(1:poolDim:end,1:poolDim:end);
    end;
end;
pooledFeatures(isnan(pooledFeatures))=0;
end

