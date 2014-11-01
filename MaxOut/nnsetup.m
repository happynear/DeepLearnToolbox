function nn = nnsetup(architecture,channel)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 0.1;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.momentumSaturate                 = 250;           %  When increase the momentum
    nn.momemtumIncrease                 = 0.75;
    nn.scaling_learningRate             = 0.5;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'softmax';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    nn.channels                         = channel;

    for i = 2 : nn.n - 1   
        % weights and weight momentum
        nn.W{i - 1} = (rand(nn.size(i) * nn.channels, nn.size(i - 1)+1) - 0.5) * sqrt(6 / (nn.size(i) + nn.size(i - 1) + nn.channels));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
    nn.W{nn.n-1} = (rand(nn.size(nn.n), nn.size(nn.n - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(nn.n) + nn.size(nn.n - 1)));
    nn.vW{nn.n-1} = zeros(size(nn.W{nn.n-1}));
end
