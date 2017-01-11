function [ nonboundSubset ] = getNonboundSubset( lagrangeMultipliers, marginParameter )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    nonboundSubset = ones(1, length(lagrangeMultipliers));
    subsetSize = 1;
    
    for sampleNumber = 1:length(lagrangeMultipliers)
       
        if ((lagrangeMultipliers(sampleNumber) ~= 0) && ...
                (lagrangeMultipliers(sampleNumber) ~= marginParameter))
            nonboundSubset(subsetSize) = sampleNumber;
            subsetSize = subsetSize + 1;
        end
    end
    
    nonboundSubset = nonboundSubset(1:(subsetSize-1));

end

