function [ value ] = RBFKernel( v1, v2, sig )

if(length(v1) == length(v2))
    sum = 0;
    for i = 1:length(v1)
        sum = sum + ((v1(i) - v2(i)) * (v1(i) - v2(i))); 
    end
    value = exp(-sum / (2 * sig * sig));
else
    value = 0;
    error('Wektory s¹ ró¿nej d³ugoœci');
end

