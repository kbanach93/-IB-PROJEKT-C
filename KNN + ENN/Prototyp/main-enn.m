clear all

% data reading
testData=csvread('testset.csv');
trainData=csvread('trainset.csv');

%data preparing
features = numel(testData)/length(testData)-1;

testLabels = testData(:, features+1);
trainLabels = trainData(:, features+1);

testData = testData(:, [1:features]);
trainData = trainData(:, [1:features]);

%normalization
testData = mat2gray(testData);
trainData = mat2gray(trainData);

%enn classification for every test sample
for testSample=1:length(testData)
    result = [];
    for k=1:2:sqrt(length(trainData))
        predictedClasses = knnClassification(testData(testSample,:), trainData, trainLabels, k);
        classes = unique(predictedClasses);
        weights = zeros(length(classes),1);
        for i=1:numel(predictedClasses)
            weight = 1/log2(i+1);
            index = find(classes == predictedClasses(i));
            weights(index) = weights(index) + weight;
        end
        [val, idx] = max(weights);
        result(size(result)+1) = classes(idx);
    end
    ennResult(testSample) = mode(result);
end

%effectiveness
effectiveness = numel(find(ennResult==transpose(testLabels)))/length(testData)
