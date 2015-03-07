function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 0.00004;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.8;          %  Momentum
    nn.scaling_learningRate             = 0.9;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    nn.epsilon                          = 1e-10;         %  numeric factor for batch normalization
    nn.useBatchNormalization            = 1;            % 

    for i = 2 : nn.n   
        
        nn.ra(i-1) = 0.2;%prelu
        nn.va(i-1) = 0;
        nn.pow(i-1) = 0.5;%pow layer
        nn.vpow(i-1) = 0;
        
        nn.W{i - 1} = randn(nn.size(i), nn.size(i - 1)+1) * sqrt(2 / (nn.size(i) + nn.size(i - 1)) / (1 + nn.ra(i-1)^2));% weights and weight momentum
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        nn.rW{i - 1} = zeros(size(nn.W{i - 1}));
        
        nn.beta{i-1} = zeros(1,nn.size(i));%batch normalization
        nn.gamma{i-1} = ones(1,nn.size(i));
        nn.sigma2{i-1} = ones(1,nn.size(i));
        nn.mu{i-1} = zeros(1,nn.size(i));
        nn.vBN{i-1} = zeros(1,nn.size(i) * 2);
        nn.rBN{i-1} = zeros(1,nn.size(i) * 2);
        nn.mean_sigma2{i-1} = zeros(1,nn.size(i));
        nn.mean_mu{i-1} = zeros(1,nn.size(i));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
    nn.ra = nn.ra(1:end-1);
    nn.va = nn.va(1:end-1);
end
