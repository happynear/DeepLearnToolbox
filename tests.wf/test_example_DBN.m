% function test_example_DBN
load mnist_uint8;
% S1=cv.FileStorage('Training Biometrika Live.yml');
% S2=cv.FileStorage('Training Biometrika Spoof.yml');
% GRAY1=S1.allPatch;
% GRAY2=S2.allPatch;
% S3=cv.FileStorage('Testing Biometrika Live.yml');
% S4=cv.FileStorage('Testing Biometrika Spoof.yml');
% GRAY3=S3.allPatch;
% GRAY4=S4.allPatch;

% train_x = [GRAY1'; GRAY2'];
% train_y = zeros(size(train_x,1),2);
% train_y(1:size(GRAY1,2),1)=1;
% train_y(size(GRAY1,2)+1:end,2) = 1;
% test_x = [GRAY3'; GRAY4'];
% test_y=train_y;
clear dbn;
clear nn;
clear opts;
clear options;

train_x = [train_real_patch';train_fake_patch'];
train_y = zeros(size(train_x,1),2);
train_y(1:size(train_real_patch,2),1)=1;
train_y(size(train_real_patch,2)+1:end,2)=1;
test_x = [test_real_patch';test_fake_patch'];
test_y=train_y;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
dbn.sizes = [8];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rng(4314);
%train dbn
dbn.sizes = [8 8];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
