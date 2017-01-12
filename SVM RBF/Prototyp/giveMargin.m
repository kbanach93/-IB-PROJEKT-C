function [ bNew ] = giveMargin( v, b, alphas, labels, data, sig )

bNew = b;
for i = 1:size(data, 1)
    bNew = bNew + alphas(i) * labels(i) * RBFKernel(v, data(i, :), sig);
end

end

