function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
%             dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) sign(nn.W{i}(:,2:end))];
        else
            dW = nn.dW{i};
        end
        
        nn.rW{i} = 0.9 * nn.rW{i} + 0.1*dW.^2;
        dW = nn.learningRate * dW ./ (sqrt(nn.rW{i})+nn.epsilon);
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
    end
    if nn.useBatchNormalization
        for i = 1 : (nn.n - 2)
            nn.rBN{i} = 0.9 * nn.rBN{i} + 0.1*nn.dBN{i}.^2;
            dBN = nn.learningRate * nn.dBN{i} ./ (sqrt(nn.rBN{i})+nn.epsilon);
            nn.vBN{i} = nn.momentum*nn.vBN{i} + dBN;
            nn.gamma{i} = nn.gamma{i} - nn.vBN{i}(1:length(nn.gamma{i}));
            nn.beta{i} = nn.beta{i} - nn.vBN{i}(length(nn.gamma{i})+1:end);
        end;
    end;
    if strcmp(nn.activation_function,'ReLU')
        da = nn.learningRate * nn.da;
        nn.va = nn.momentum*nn.va + da;
        nn.ra = nn.ra - nn.va;
    end;
    
end
