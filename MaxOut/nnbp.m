function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear'}
            d{n} = - nn.e;
    end
    for i = (n - 1) : -1 : 2
        
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = d{i + 1} * nn.W{i}; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = d{i + 1}(:,2:end) * nn.W{i};
        end
        
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end
        
        de = kron(d{i}(:,2:end),ones(1,nn.channels));
%         de = repmat(d{i}(:,2:end),1,nn.channels);
        d{i} = [d{i}(:,1) de .* nn.maxMask{i}];

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
end
