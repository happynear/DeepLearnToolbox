function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;

    %feedforward pass
    for i = 2 : n-1
        nn.a{i} = nn.a{i - 1} * nn.W{i - 1}';
        nn.a{i} = reshape(nn.a{i},[m, nn.channels, nn.size(i)]);
        [nn.a{i}, maxp] = max(nn.a{i},[],2);
        maxp = squeeze(maxp);
        maxp = maxp + repmat((0:nn.size(i)-1)*nn.channels,m,1);
        [I J V] = find(maxp);
        nn.maxMask{i} = sparse(I,V,ones(length(V),1),m,nn.size(i) * nn.channels);
        nn.a{i} = squeeze(nn.a{i});
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
%       size(y)
%        size(nn.a{n})
     nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
