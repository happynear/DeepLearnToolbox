%function test_example_SAE
%load mnist_uint8;
% S1=cv.FileStorage('Training Biometrika Live.yml');
% S2=cv.FileStorage('Training Biometrika Spoof.yml');
% GRAY1=S1.allPatch;
% GRAY2=S2.allPatch;
% S3=cv.FileStorage('Testing Biometrika Live.yml');
% S4=cv.FileStorage('Testing Biometrika Spoof.yml');
% GRAY3=S3.allPatch;
% GRAY4=S4.allPatch;

% GRAY1=Trans*double(GRAY1);
% GRAY2=Trans*double(GRAY2);

% train_x = [GRAY1'; GRAY2'];
% train_y = zeros(size(train_x,1),2);
% train_y(1:size(GRAY1,2),1)=1;
% train_y(size(GRAY1,2)+1:end,2) = 1;
% test_x = [GRAY3'; GRAY4'];
% test_y=train_y;

train_x=[train_real_patch train_fake_patch]';
test_x=[test_real_patch test_fake_patch]';
train_y=zeros(size(train_real_patch,2)+size(train_fake_patch,2),2);
train_y(1:size(train_real_patch,2),1)=1;
train_y(size(train_real_patch,2)+1:size(train_real_patch,2)*2,2)=1;
test_y=train_y;

train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

layer1=size(train_x,2);
layer2=12;

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([layer1 layer2]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.16;
opts.numepochs =   1;
opts.batchsize = 50;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([layer1 layer2 2]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 50;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
er
%assert(er < 0.25, 'Too big error');

%% ex2 train a 100-100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([layer1 layer2 layer2]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.16;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.16;

opts.numepochs =   1;
opts.batchsize = 50;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([layer1 layer2 layer2 2]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;

%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 50;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
er
assert(er < 0.25, 'Too big error');
