% function test_example_NN
% load mnist_uint8;
close all;
% trainNum = 1000;
% idx = randperm(2000);
% images = double(images);
% images = bsxfun(@rdivide, images, sum(images,2));
% train_x = images(idx(1:trainNum),:);
% labels(labels==0)=10;
% groundTruth = full(sparse(labels(1:2000), 1:2000, 1));
% train_y = groundTruth(:,idx(1:trainNum))';
% test_x = images(idx(trainNum+1:2000),:);
% test_y = groundTruth(:,idx(trainNum+1:2000))';

% S = cv.FileStorage('E:\LearnOpenCV\nn\data2.yml');
% test_x = S.images(1:2000,:);
% labels = S.labels;
% labels(labels==0)=10;
% groundTruth = full(sparse(double(labels(1:2000)), 1:2000, 1));
% test_y = groundTruth(:,1:2000)';
% 
% S = cv.FileStorage('E:\LearnOpenCV\nn\data1.yml');
% num = floor(size(S.images,1)/100)*100;
% train_x = S.images(1:num,:);
% labels = S.labels;
% labels(labels==0)=10;
% groundTruth = full(sparse(double(labels(1:num)), 1:num, 1));
% train_y = groundTruth(:,1:num)';

% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
train_x = training;
train_y = [group_train==-1 group_train==1];
test_x = sample;
test_y  = [group_sample==-1 group_sample==1];
% train_x = [train_x;test_x];
% train_y = [train_y;test_y];

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);
% all_x = normalize(images, mu, sigma);
% all_y = full(sparse(labels, 1:length(labels), 1))';
clear nn;

% % ex1 vanilla neural net
% rand('state',0)
% nn = nnsetup([720 100 10]);
% opts.numepochs =  1;   %  Number of full sweeps through data
% opts.batchsize = 100;  %  Take a mean gradient step over this many samples
% [nn, L] = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% 
% disp(['rate: ', num2str((1-er)*100) '%']);
% % assert(er < 0.08, 'Too big error');
% 
% %% ex2 neural net with L2 weight decay
% rand('state',0)
% nn = nnsetup([720 100 10]);
% 
% nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% opts.numepochs =  1;        %  Number of full sweeps through data
% opts.batchsize = 100;       %  Take a mean gradient step over this many samples
% 
% nn = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% disp(['rate: ', num2str((1-er)*100) '%']);
% % assert(er < 0.1, 'Too big error');
% 
% 
% %% ex3 neural net with dropout
% rand('state',0)
% nn = nnsetup([720 100 10]);
% 
% nn.dropoutFraction = 0.5;   %  Dropout fraction 
% opts.numepochs =  1;        %  Number of full sweeps through data
% opts.batchsize = 100;       %  Take a mean gradient step over this many samples
% 
% nn = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% disp(['rate: ', num2str((1-er)*100) '%']);
% % assert(er < 0.1, 'Too big error');
% 
% %% ex4 neural net with sigmoid activation function
% rand('state',0)
% nn = nnsetup([720 100 10]);
% 
% nn.activation_function = 'sigm';    %  Sigmoid activation function
% nn.learningRate = 1;                %  Sigm require a lower learning rate
% opts.numepochs =  1;                %  Number of full sweeps through data
% opts.batchsize = 100;               %  Take a mean gradient step over this many samples
% 
% nn = nntrain(nn, train_x, train_y, opts);
% 
% [er, bad] = nntest(nn, test_x, test_y);
% disp(['rate: ', num2str((1-er)*100) '%']);
% % assert(er < 0.1, 'Too big error');
% S = struct('W1', nn.W{1}', 'W2', nn.W{2}','mu', mu, 'sigma', sigma);
% cv.FileStorage('weight.yml',S);

%% ex5 plotting functionality
rand('state',0)
nn = nnsetup([4096 100 10 2]);
opts.numepochs         = 30;            %  Number of full sweeps through data
nn.activation_function = 'ReLU';
nn.output              = 'softmax';    %  use softmax output
opts.batchsize         = 100;         %  Take a mean gradient step over this many samples
opts.plot              = 1;            %  enable plotting
% nn.nonSparsityPenalty = 0.1;
nn.dropoutFraction   = 0.5;
% nn.inputZeroMaskedFraction          = 0.6; 

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
disp(['rate: ', num2str((1-er)*100) '%']);
assert(er < 0.1, 'Too big error');
S = struct('W1', nn.W{1}', 'W2', nn.W{2}','mu', mu, 'sigma', sigma);
cv.FileStorage('weight.yml',S);
% [er, bad] = nntest(nn, all_x, all_y);
%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data
% vx   = train_x(1:10000,:);
% tx = train_x(10001:end,:);
% vy   = train_y(1:10000,:);
% ty = train_y(10001:end,:);
% 
% rand('state',0)
% nn                      = nnsetup([784 20 10]);     
% nn.output               = 'softmax';                   %  use softmax output
% opts.numepochs          = 5;                           %  Number of full sweeps through data
% opts.batchsize          = 1000;                        %  Take a mean gradient step over this many samples
% opts.plot               = 1;                           %  enable plotting
% nn = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)
% 
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.1, 'Too big error');
