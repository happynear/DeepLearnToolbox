
layers = randi(30,1,10);

nn = nnsetup([2 layers 3]);
nn.activation_function              = 'ReLU';
nn.output = 'sigm';
nn.useBatchNormalization            = 1;

output_h = 600;
output_w = 800;

[I,J] = ind2sub([output_h,output_w],(1:output_h*output_w)');

nn = nnff(nn,[I J],zeros(size(I,1),3));

output = nn.a{length(nn.a)};
output = zscore(output);
output = reshape(output,[output_h,output_w,3]);

imshow(uint8(output*255));