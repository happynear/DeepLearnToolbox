% function test_example_CNN
load mnist_uint8;
global useGpu;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)
clear cnn;
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 6, 'activation', 'ReLU') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'm') %sub sampling layer 'm':maxpooling; 'a':average pooling; 's':stochastic pooling
%     struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5, 'activation', 'ReLU') %convolution layer
%     struct('type', 's', 'scale', 2, 'method', 'm') %subsampling layer
};
debug =true;
if debug
    opts.alpha = 1;
    opts.batchsize = 10;
    opts.numepochs = 1;
    cnn = cnnsetup(cnn, train_x(:,:,1:10), train_y(:,1:10));
%     cnn = cnntrain(cnn, train_x(:,:,1:10), train_y(:,1:10), opts);
    cnnnumgradcheck(cnn, train_x(:,:,1:10), train_y(:,1:10));
end;

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
