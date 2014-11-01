function net = cnntrain(net, x, y, opts)
    global useGpu;
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
%         tic;
        kk = randperm(m);
        for l = 1 : numbatches
            if useGpu
                batch_x = gpuArray(x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)));
                batch_y = gpuArray(y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)));
            else
                batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
                batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            end;
            
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = gather(net.L);
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * gather(net.L);
            if mod(l,10)==0
                disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) 'batch ' num2str(l) '/' num2str(numbatches)]);
            end;
        end
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ' error:' num2str(net.L)]);
%         toc;
    end
    
end
