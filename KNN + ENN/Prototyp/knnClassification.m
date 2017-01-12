function [ out ] = knnClassification(testData, trainData, trainLabels, k)

dist = pdist2(testData, trainData);
  for i=1:k
   [val(i),idx] = min(dist);
   out(i) = trainLabels(idx);
   dist(idx) = max(dist);
  end
end

