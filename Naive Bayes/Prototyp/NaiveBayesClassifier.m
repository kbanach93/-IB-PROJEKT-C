function [statistic,effect] = NaiveBayesClassifier(data)
%data - array with values of features and class value in the last
%column
%statistic matrix with correct and incorrect classified object
%conf - efectivness of classification

input_data = [data];
input_features = double(input_data(:,1:4)); % matrix with features
input_classes = double(input_data(:,6)); % matrix with classes

% Creating radom training and testing group
random_data = cvpartition(input_classes,'holdout',.2);

% Training group
train_features = input_features(training(random_data,1),:);
train_classes = input_classes(training(random_data,1));

% Testing grop
test_features=input_features(test(random_data,1),:);
test_classes=input_classes(test(random_data,1),:);

unique_classes=unique(train_classes); % vector of unique classes
number_of_classes=length(unique_classes); % number of classes
number_of_features=size(train_features,2); % number of features
number_of_data_test=length(test_classes); % number if data in testing data set

% Incidence of class
for i=1:number_of_classes
    fy(i)=sum(double(train_classes==unique_classes(i)))/length(train_classes);
end

for i=1:number_of_classes
      xi=train_features((train_classes==unique_classes(i)),:);
      average(i,:)=mean(xi,1); %average of feature according to classes
      sigma(i,:)=std(xi,1); % standard deviaton of fetures according to classes
end

% Probability for each sample in test class
for j=1:number_of_data_test
    pdf=normcdf(ones(number_of_classes,1)*test_features(j,:),average,sigma); %funkcja gestosci prawdoppodobienstwa
    Probability(j,:)=fy.*prod(pdf,2)'; % Probability for each sample according to Bayas probability
end

%Result of classifier
[pv0,id]=max(Probability,[],2);
for i=1:length(id)
    predict_class(i,1)=unique_classes(id(i));
end

% Compare predicted output with actual output from test data
 statistic = [];
 
 unique_test_classes = unique(test_classes);
 for i=1:length(unique_test_classes)
    for j=1:length(unique_test_classes)
        statistic(i,j) = sum(test_classes==unique_test_classes(i) & predict_class == unique_test_classes(j));
    end
 end
 statistic
 
effect=sum(predict_class==test_classes)/length(predict_class)