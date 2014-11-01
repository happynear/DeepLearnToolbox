function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;
    global useGpu;

    for l = 2 : n-1   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                if useGpu
                    z = gpuArray.zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                else
                    z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                end;
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end;
                %  add bias, pass through nonlinearity
                if strcmp(net.layers{l}.activation, 'sigmoid')
                    net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
                elseif strcmp(net.layers{l}.activation, 'tanh')
                    net.layers{l}.a{j} = tanh(z + net.layers{l}.b{j});% need to be exploited
                elseif strcmp(net.layers{l}.activation, 'ReLU')
                    net.layers{l}.a{j} = max(z + net.layers{l}.b{j},0);
                end;
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            if net.testing && strcmp(net.layers{l}.method, 's')
                for j = 1 : inputmaps
                    net.layers{l}.a{j} = StochaticTest(net.layers{l}.scale, net.layers{l - 1}.a{j});
                end;
            else
                %  downsample
                if strcmp(net.layers{l}.method, 'a')
                    for j = 1 : inputmaps
                        z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                        net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                        net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                        net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                    end;
                elseif strcmp(net.layers{l}.method, 'm')
                    for j = 1 : inputmaps
                        if useGpu
                            [zz, maxPosition] = MaxPooling(gather(net.layers{l - 1}.a{j}),[net.layers{l}.scale net.layers{l}.scale]);%this is stupid. I don't know how to code CUDA parellel program so far.
                            net.layers{l}.a{j} = gpuArray(zz);
                        else
                            [net.layers{l}.a{j}, maxPosition] = MaxPooling(net.layers{l - 1}.a{j},[net.layers{l}.scale net.layers{l}.scale]);
                        end;
                        maxPosition = sparse(ones(length(maxPosition),1),maxPosition,ones(length(maxPosition),1),1,numel(net.layers{l - 1}.a{j}));
                        net.layers{l}.PosMatrix{j} = reshape(full(maxPosition),size(net.layers{l - 1}.a{j})); % matlab不支持多维稀疏矩阵- -
                        net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                        net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                    end;
                elseif strcmp(net.layers{l}.method, 's')
                    for j = 1 : inputmaps
                        if useGpu
                            [zz, maxPosition] = StochasticPooling(gather(net.layers{l - 1}.a{j}),[net.layers{l}.scale net.layers{l}.scale]);%this is stupid
                            net.layers{l}.a{j} = gpuArray(zz);
                        else
                            [net.layers{l}.a{j}, maxPosition] = StochasticPooling(net.layers{l - 1}.a{j},[net.layers{l}.scale net.layers{l}.scale]);
                        end;
                        maxPosition = sparse(ones(length(maxPosition),1),maxPosition,ones(length(maxPosition),1),1,numel(net.layers{l - 1}.a{j}));
                        net.layers{l}.PosMatrix{j} = reshape(full(maxPosition),size(net.layers{l - 1}.a{j})); % matlab不支持多维稀疏矩阵- -
                        net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                        net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                    end;
                elseif strcmp(net.layers{l}.method, 'o')
                    net.layers{l}.PosMatrix = zeros(size(net.layers{l - 1}.a{1}));
                    
                end;
            
            end;
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n-1}.a)
        sa = size(net.layers{n-1}.a{j});
        net.fv = [net.fv; reshape(net.layers{n-1}.a{j}, sa(1) * sa(2), sa(3))];
    end;
    %  feedforward into output perceptrons
    if strcmp(net.layers{n}.objective, 'sigm')
        net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
    elseif strcmp(net.layers{n}.objective, 'softmax')
        M = net.ffW*net.fv;
        M = bsxfun(@plus, M, net.ffb);
        M = bsxfun(@minus, M, max(M, [], 1));
        M = exp(M);
        M = bsxfun(@rdivide, M, sum(M));
        net.o = M;
    end;

end
