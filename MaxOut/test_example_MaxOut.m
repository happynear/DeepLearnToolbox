% function test_example_MaxOut
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

debug = false;
if debug
    rand('state',0)
    nn                      = nnsetup([21 5 10],5);     
    nn.output               = 'softmax';                   %  use softmax output
    % nn.dropoutFraction      = 0.5;
    opts.numepochs          = 3;                           %  Number of full sweeps through data
    opts.batchsize          = 100;                        %  Take a mean gradient step over this many samples
    opts.plot               = 0;                           %  enable plotting
    nnchecknumgrad(nn,randn(100,21),train_y(1:100,:));
end;


rand('state',0)
nn                      = nnsetup([784 240 240 10],5);     
nn.output               = 'softmax';                   %  use softmax output
nn.dropoutFraction      = 0.8;
opts.numepochs          = 3;                           %  Number of full sweeps through data
opts.batchsize          = 100;                        %  Take a mean gradient step over this many samples
opts.plot               = 0;                           %  enable plotting
nn = nntrain(nn, train_x, train_y, opts);                %  nntrain takes validation set as last two arguments (optionally)

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');
