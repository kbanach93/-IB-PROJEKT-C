function [ decision ] = svmPredict( v, b, alphas, labels, data, sig )

outcome = giveMargin(v, b, alphas, labels, data, sig);
if(outcome > 0)
    decision = 1;
else
    decision = -1;
end


end

