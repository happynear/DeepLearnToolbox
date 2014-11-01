function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    nn.iter = nn.iter + 1;
    if nn.iter == nn.momentumSaturate
        nn.momemtum = nn.momemtumIncrease;
    end;
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        else
            dW = nn.dW{i};
        end
        
        dW = nn.learningRate * dW;
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
        nn.W{i} = bsxfun(@rdivide, nn.W{i},sqrt(sum(nn.W{i}.^2))*1.9365);
    end
end
