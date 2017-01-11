function [ randomizedIndices ] = randomizeIndexOrder( dataSet )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    startPoint = randi(length(dataSet));
    
    if (startPoint == 1)
        randomizedIndices = 1:length(dataSet);
    else
        randomizedIndices = [dataSet(startPoint:length(dataSet)), dataSet(1:(startPoint-1))];
    end
end

