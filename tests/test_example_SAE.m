% function test_example_SAE
% load mnist_uint8;

S = cv.FileStorage('E:\LearnOpenCV\nn\data2.yml');
num = floor(size(S.images,1)/100)*100;
test_x = S.images(1:num,:);
test_x = bsxfun(@rdivide,test_x,sum(test_x,2));
% test_x = reshape(test_x,num,30,24);
labels = S.labels;
labels(labels==0)=10;
groundTruth = full(sparse(double(labels(1:num)), 1:num, 1));
test_y = groundTruth(:,1:num)';

S = cv.FileStorage('E:\LearnOpenCV\nn\data1.yml');
num = floor(size(S.images,1)/100)*100;
train_x = S.images(1:num,:);
train_x = bsxfun(@rdivide,train_x,sum(train_x,2));
% train_x = reshape(train_x,num,30,24);
labels = S.labels;
labels(labels==0)=10;
groundTruth = full(sparse(double(labels(1:num)), 1:num, 1));
train_y = groundTruth(:,1:num)';

% train_x = double(train_x)/255;
% test_x  = double(test_x)/255;
% train_y = double(train_y);
% test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([784 100]);
sae.ae{1}.activation_function       = 'ReLU';
sae.ae{1}.output       = 'ReLU';
% sae.ae{1}.inputZeroMaskedFraction   = 0.5;
% sae.ae{1}.nonSparsityPenalty   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')
sum(sae.ae{1}.a{1}(:,2:end)>0,2)

% Use the SDAE to initialize a FFNN
nn = nnsetup([720 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
opts.numepochs =   50;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.16, 'Too big error');
