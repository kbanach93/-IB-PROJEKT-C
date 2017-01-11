function [ flagExamineSuccess, sampleError, lagrangeMultipliers, bias ] = examineSample( pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter, tolerance )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
    flagExamineSuccess = 0;
    flagOptimizeSuccess = 0;
	sampleError(pairSecond) = sum(lagrangeMultipliers.*labelSet.*gramMatrix(:,pairSecond)) - bias - labelSet(pairSecond);
        
	if ((labelSet(pairSecond)*sampleError(pairSecond) < -tolerance && ...
        lagrangeMultipliers(pairSecond) < marginParameter) || ...
        (labelSet(pairSecond)*sampleError(pairSecond) > tolerance && ...
        lagrangeMultipliers(pairSecond) > 0))           
        
        nonboundSubset = getNonboundSubset(lagrangeMultipliers, marginParameter);
        
        if (length(nonboundSubset) > 1)
            % stay with loop for easier C++ implementation
            maximalError = 0;
            pairFirst = 1;

            for evaluatedSample = 1:length(lagrangeMultipliers)
                if(pairSecond ~= evaluatedSample) %ensures samples are different
                    sampleError(evaluatedSample) = sum(lagrangeMultipliers.*labelSet.*gramMatrix(:,evaluatedSample)) - bias - labelSet(evaluatedSample);
                    if(abs(sampleError(pairSecond) - sampleError(evaluatedSample)) > maximalError) %finds maximal error
                        maximalError = abs(sampleError(pairSecond) - sampleError(evaluatedSample));
                        pairFirst = evaluatedSample; %saves index of the maximal error sample
                    end
                end
            end  

            [flagOptimizeSuccess, lagrangeMultipliers, bias, sampleError] = optimizePair( pairFirst, pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter);

            if (flagOptimizeSuccess == 1)
                flagExamineSuccess = 1;
                return
            else
                flagExamineSuccess = 0;
            end
       
        
            for pairFirst = randomizeIndexOrder(nonboundSubset);
                
                [flagOptimizeSuccess, lagrangeMultipliers, bias, sampleError] = optimizePair( pairFirst, pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter);

                if (flagOptimizeSuccess == 1)
                    flagExamineSuccess = 1;
                    return
                else
                    flagExamineSuccess = 0;
                end 
            end
        end
        
        for pairFirst = randomizeIndexOrder(1:length(lagrangeMultipliers))

            [flagOptimizeSuccess, lagrangeMultipliers, bias, sampleError] = optimizePair( pairFirst, pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter);

            if (flagOptimizeSuccess == 1)
                flagExamineSuccess = 1;
                return
            else
                flagExamineSuccess = 0;
            end
        end        
	end      
end

