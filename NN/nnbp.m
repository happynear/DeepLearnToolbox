function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear', 'ReLU'}
            d{n} = - nn.e;
    end
    for i = (n - 1) : -1 : 2
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
%             sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * ones(size(pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError); % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError);
        end
        
        if(nn.dropoutFraction>0&&i>=n-3)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end
        
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
            case 'ReLU'
                d_act = (nn.a{i} > 0) + nn.ra(i-1) * (nn.a{i} < 0);
                tt = d{i}(nn.a{i} < 0) .* nn.a{i}(nn.a{i}<0) / nn.ra(i-1);
%                 tt = tt(nn.a{i} < 0);
                nn.da(i-1) = sum(tt) / size(nn.a{i},1);
            case 'linear'
                d_act = ones(size(nn.a{i}));
        end
        d{i} = d{i} .* d_act;

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
end
