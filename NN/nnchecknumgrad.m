function nnchecknumgrad(nn, x, y)
    epsilon = 1e-4;
    er = 1e-7;
    n = nn.n;
    rng('default');
    nnr = nn;
    nnr = nnff(nnr, x, y);
    nnr = nnbp(nnr);
%     for l  = 1 : (n - 2)
%         nn_m = nn; nn_p = nn;
%         nn_m.ra(l) = nn.ra(l) - epsilon;
%         nn_p.ra(l) = nn.ra(l) + epsilon;
%         rng('default');
%         nn_m = nnff(nn_m, x, y);
%         rng('default');
%         nn_p = nnff(nn_p, x, y);
%         dW = (nn_p.L - nn_m.L) / (2 * epsilon);
%         rdW = nnr.da(l);
%         e = abs(dW - rdW);
% 
%         assert(e < er, 'numerical gradient checking failed');
%     end;
%     for l  = 1 : (n - 2)
%         for j = 1 : size(nn.W{l}, 2)-1
%             nn_m = nn; nn_p = nn;
%             nn_m.gamma{l}(j) = nn.gamma{l}(j) - epsilon;
%             nn_p.gamma{l}(j) = nn.gamma{l}(j) + epsilon;
%             rng('default');
%             nn_m = nnff(nn_m, x, y);
%             rng('default');
%             nn_p = nnff(nn_p, x, y);
%             dW = (nn_p.L - nn_m.L) / (2 * epsilon);
%             rdW = nnr.dBN{l}(j);
%             e = abs(dW - rdW);
% 
%             assert(e < er, 'numerical gradient checking failed');
%         end;
%     end;

    for l = 1 : (n - 1)
        for i = 1 : size(nn.W{l}, 1)
            for j = 1 : size(nn.W{l}, 2)
                nn_m = nn; nn_p = nn;
                nn_m.W{l}(i, j) = nn.W{l}(i, j) - epsilon;
                nn_p.W{l}(i, j) = nn.W{l}(i, j) + epsilon;
                rng('default');
                nn_m = nnff(nn_m, x, y);
                rng('default');
                nn_p = nnff(nn_p, x, y);
                dW = (nn_p.L - nn_m.L) / (2 * epsilon);
                rdW = nnr.dW{l}(i, j);
                e = abs(dW - rdW);
                
%                 assert(e < er, 'numerical gradient checking failed');
            end
        end
    end
end
