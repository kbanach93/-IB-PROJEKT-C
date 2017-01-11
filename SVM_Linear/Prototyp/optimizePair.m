function [ flagOptimizeSuccess, lagrangeMultipliers, bias, sampleError ] = optimizePair( pairFirst, pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


    roundingError = 1e-10;

    flagOptimizeSuccess = 0;
    sampleError(pairFirst) = sum(lagrangeMultipliers.*labelSet.*gramMatrix(:,pairFirst)) - bias - labelSet(pairFirst);
    sampleError(pairSecond) = sum(lagrangeMultipliers.*labelSet.*gramMatrix(:,pairSecond)) - bias - labelSet(pairSecond);

    if (pairFirst == pairSecond)
        flagOptimizeSuccess = 0;
        return
    end

    if (labelSet(pairSecond) == labelSet(pairFirst)),
        lowBoundary = max(0, lagrangeMultipliers(pairFirst) + lagrangeMultipliers(pairSecond) - marginParameter);
        highBoundary = min(marginParameter, lagrangeMultipliers(pairFirst) + lagrangeMultipliers(pairSecond));
    else
        lowBoundary = max(0, lagrangeMultipliers(pairSecond) - lagrangeMultipliers(pairFirst));
        highBoundary = min(marginParameter, marginParameter + lagrangeMultipliers(pairSecond) - lagrangeMultipliers(pairFirst));
    end

    if (lowBoundary == highBoundary),
        flagOptimizeSuccess = 0;
        return
    end

    secondDerivative = gramMatrix(pairSecond,pairSecond) + gramMatrix(pairFirst,pairFirst) - 2*gramMatrix(pairFirst,pairSecond);

    if (secondDerivative > 0)

        newSecondLagrangeMultiplier = lagrangeMultipliers(pairSecond) + (labelSet(pairSecond)*(sampleError(pairFirst) - sampleError(pairSecond))/secondDerivative);

        if (newSecondLagrangeMultiplier < lowBoundary)
            newSecondLagrangeMultiplier = lowBoundary;
        elseif (newSecondLagrangeMultiplier > highBoundary)
            newSecondLagrangeMultiplier = highBoundary;
        end

    else

        s = labelSet(pairFirst)*labelSet(pairSecond);
        firstFunction = labelSet(pairFirst)*(sampleError(pairFirst) + bias) - lagrangeMultipliers(pairFirst)*gramMatrix(pairFirst,pairFirst) - s*lagrangeMultipliers(pairSecond)*gramMatrix(pairFirst,pairSecond);
        secondFunction = labelSet(pairSecond)*(sampleError(pairSecond) + bias) - s*lagrangeMultipliers(pairFirst)*gramMatrix(pairFirst,pairSecond) - lagrangeMultipliers(pairSecond)*gramMatrix(pairSecond,pairSecond);
        lowFunction = lagrangeMultipliers(pairFirst) + s*(lagrangeMultipliers(pairSecond) - lowBoundary);
        highFunction = lagrangeMultipliers(pairFirst) + s*(lagrangeMultipliers(pairSecond) - highBoundary);

        lowObjectiveFunction = lowFunction*firstFunction + lowBoundary*secondFunction + 0.5*(lowFunction*lowFunction)*gramMatrix(pairFirst,pairFirst) + 0.5*lowBoundary*lowBoundary*gramMatrix(pairSecond,pairSecond) + s*lowBoundary*lowFunction*gramMatrix(pairFirst,pairSecond);
        highObjectiveFunction = highFunction*firstFunction + highBoundary*secondFunction + 0.5*(highFunction*highFunction)*gramMatrix(pairFirst,pairFirst) + 0.5*highBoundary*highBoundary*gramMatrix(pairSecond,pairSecond) + s*highBoundary*highFunction*gramMatrix(pairFirst,pairSecond);

        if (lowObjectiveFunction < (highObjectiveFunction - roundingError))
            newSecondLagrangeMultiplier = lowBoundary;
        elseif (lowObjectiveFunction > (highObjectiveFunction + roundingError))
            newSecondLagrangeMultiplier = highBoundary;
        else
            newSecondLagrangeMultiplier = lagrangeMultipliers(pairSecond);
        end
    end

    if (abs(newSecondLagrangeMultiplier - lagrangeMultipliers(pairSecond)) < roundingError*(newSecondLagrangeMultiplier + lagrangeMultipliers(pairSecond) + roundingError))
        flagOptimizeSuccess = 0;
        return
    end

    newFirstLagrangeMultiplier = lagrangeMultipliers(pairFirst) + labelSet(pairSecond)*labelSet(pairFirst)*(lagrangeMultipliers(pairSecond) - newSecondLagrangeMultiplier);

    pairSecondBias = sampleError(pairSecond) + labelSet(pairSecond)*(newFirstLagrangeMultiplier - lagrangeMultipliers(pairFirst))*gramMatrix(pairFirst,pairSecond) + labelSet(pairSecond)*(newSecondLagrangeMultiplier - lagrangeMultipliers(pairSecond))*gramMatrix(pairSecond,pairSecond) + bias;
    pairFirstBias = sampleError(pairFirst) + labelSet(pairSecond)*(newFirstLagrangeMultiplier - lagrangeMultipliers(pairFirst))*gramMatrix(pairFirst,pairFirst) + labelSet(pairSecond)*(newSecondLagrangeMultiplier - lagrangeMultipliers(pairSecond))*gramMatrix(pairFirst,pairSecond) + bias;

    lagrangeMultipliers(pairFirst) = newFirstLagrangeMultiplier;
    lagrangeMultipliers(pairSecond) = newSecondLagrangeMultiplier;

    if (0 < lagrangeMultipliers(pairSecond) && lagrangeMultipliers(pairSecond) < marginParameter),
        bias = pairSecondBias;
    elseif (0 < lagrangeMultipliers(pairFirst) && lagrangeMultipliers(pairFirst) < marginParameter),
        bias = pairFirstBias;
    else
        bias = (pairSecondBias+pairFirstBias)/2;
    end

    flagOptimizeSuccess = 1;
end

