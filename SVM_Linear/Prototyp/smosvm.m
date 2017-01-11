function [ w, bias ] = smosvm(trainSet, labelSet, marginParameter)
%This function finds optimal hyperplane separating the values in trainSet
%based on labelSet and marginParameter value. To do so it uses Sequential Minimal
%Optimization

    sampleCount = size(trainSet, 1); %equals Lagrange multipliers count too
    lagrangeMultipliers = zeros(sampleCount, 1);
    sampleError = zeros(sampleCount, 1);
    bias = 0;
    tolerance = 0.001;
    flagExamineAll = 1;
    flagHasChanged = 0;
    flagExamineSuccess = 0;
    gramMatrix = trainSet*trainSet'; %defines linear SVM, can be modified
    
	while (flagExamineAll == 1 || flagHasChanged == 1)
        
        flagHasChanged = 0;
        
        if (flagExamineAll ==1)
        
            for pairSecond = 1:sampleCount
                
                [ flagExamineSuccess, sampleError, lagrangeMultipliers, bias ] = examineSample( pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter, tolerance );
                if (flagExamineSuccess == 1)
                    flagHasChanged = 1;
                end
            end
            
        else

            for pairSecond = getNonboundSubset(lagrangeMultipliers, marginParameter)
                
                [ flagExamineSuccess, sampleError, lagrangeMultipliers, bias ] = examineSample( pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter, tolerance );
                if (flagExamineSuccess == 1)
                    flagHasChanged = 1;
                end
            end
        end
        
        if (flagExamineAll == 1)
            flagExamineAll = 0;
        elseif (flagHasChanged == 0)
            flagExamineAll = 1;
        end
	end

    w = ((lagrangeMultipliers.*labelSet)'*trainSet)';

end