% function test_example_DBN
% load mnist_uint8;

% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);
% S = cv.FileStorage('E:\LearnOpenCV\nn\data2.yml');
% test_x = S.images(1:2000,:);
% test_x = reshape(test_x,2000,30,24);
% labels = S.labels;
% labels(labels==0)=10;
% groundTruth = full(sparse(double(labels(1:2000)), 1:2000, 1));
% test_y = groundTruth(:,1:2000)';
% 
% S = cv.FileStorage('E:\LearnOpenCV\nn\trans1400.yml');
% train_x = S.images(1:1400,:);
% train_x = reshape(train_x,1400,30,24);
% labels = S.labels;
% labels(labels==0)=10;
% groundTruth = full(sparse(double(labels(1:1400)), 1:1400, 1));
% train_y = groundTruth(:,1:1400)';

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 50 10 10];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';
nn.output              = 'softmax';    %  use softmax output
nn.dropoutFraction   = 0.5;

%train nn
opts.numepochs =  30;
opts.batchsize = 100;
opts.plot = 1;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
