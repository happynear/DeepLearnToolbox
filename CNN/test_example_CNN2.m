% function test_example_CNN
% load mnist_uint8;
global useGpu;
useGpu = false;

trainNum = 700;
idx = randperm(1400);
images = double(images);
images = bsxfun(@rdivide, images, max(images,[],2));
train_x = images(idx(1:trainNum),:);
labels(labels==0)=10;
groundTruth = full(sparse(labels(1:1400), 1:1400, 1));
train_y = groundTruth(:,idx(1:trainNum))';
test_x = images(idx(trainNum+1:1400),:);
test_y = groundTruth(:,idx(trainNum+1:1400))';
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
train_x = double(reshape(train_x',30,24,700))/255;
test_x = double(reshape(test_x',30,24,700))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

if exist('MaxPooling')~=3
    mex MaxPooling.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -largeArrayDims 
end;
if exist('StochasticPooling')~=3
    mex StochasticPooling.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -largeArrayDims 
end;

rand('state',0)
clear cnn;
% cnn.layers = {
%     struct('type', 'i') %input layer
%     struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9, 'activation', 'ReLU') %convolution layer
%     struct('type', 's', 'scale', 2, 'method', 's') %sub sampling layer 'm':maxpooling; 'a':average pooling; 's':stochastic pooling
% %     struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5, 'activation', 'ReLU') %convolution layer
% %     struct('type', 's', 'scale', 2, 'method', 'm') %subsampling layer
%     struct('type','o','objective','softmax');
% };
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5, 'activation', 'sigmoid') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'm') %sub sampling layer 'm':maxpooling; 'a':average pooling; 's':stochastic pooling
%     struct('type', 'c', 'outputmaps', 12, 'kernelsize', 2, 'activation', 'ReLU') %convolution layer
%     struct('type', 's', 'scale', 2, 'method', 'm') %subsampling layer
    struct('type','o','objective','softmax');
};
debug =false;
if debug
    opts.alpha = 1;
    opts.batchsize = 10;
    opts.numepochs = 1;
    cnn = cnnsetup(cnn, train_x(:,:,1:10), train_y(:,1:10));
%     cnn = cnntrain(cnn, train_x(:,:,1:10), train_y(:,1:10), opts);
    cnnnumgradcheck(cnn, train_x(:,:,1:10), train_y(:,1:10));
end;

opts.alpha = 0.1;
opts.alphascale = 0.5; 
opts.batchsize = 50;
opts.numepochs = 100;
opts.momentum = 0.95;
opts.momIncrease = 20;
cnn.iter = 1;
cnn.testing = false;
cnn.dropoutFraction = 0.5;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
