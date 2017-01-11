 #include <iostream>
 #include <Eigen/Dense>
 #include <cstdlib>
 #include <ctime>
 #include <fstream>

 using Eigen::MatrixXd;
 using Eigen::VectorXd;
 using namespace std;

 double getBiggerNumber(double firstNumber, double secondNumber)
 {
 	if (firstNumber>secondNumber)
 	{
 		return firstNumber;
 	}
 	else
 	{
 		return secondNumber;
 	}
 }

 double getSmallerNumber(double firstNumber, double secondNumber)
 {
 	if (firstNumber<secondNumber)
 	{
 		return firstNumber;
 	}
 	else
 	{
 		return secondNumber;
 	}
 }

 void random_data_generator(MatrixXd &trainSet, MatrixXd &labelSet)
 {
     double max1 = 5.0;
     double max2 = 10.0;
 	trainSet << ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)),
 		((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)),
 		((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)), ((double) rand()*max1 / (RAND_MAX)),
 		((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max1 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0,
 		((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max1 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0,
 		((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max1 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0, ((double) rand()*max2 / (RAND_MAX)) + 4.0;

 	labelSet << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1;
 }

 void randomizeIndexOrder(MatrixXd dataSet, MatrixXd &randomizedIndices)
 {
 	VectorXd randomizedIndices2;
 	VectorXd vec1;
 	VectorXd vec2;

 	int startPoint = rand() % dataSet.size();

 	if (startPoint == 0)
 	{
 		randomizedIndices2 = VectorXd::LinSpaced(dataSet.size(), 0, dataSet.size() - 1);
 	}
 	else
 	{
 		randomizedIndices2 = VectorXd::LinSpaced(dataSet.size(), 0, dataSet.size() - 1);
 		vec1 = randomizedIndices2.block(startPoint, 0, dataSet.size() - startPoint, 1);
 		vec2 = randomizedIndices2.block(0, 0, startPoint, 1);
 		randomizedIndices2 << vec1, vec2;
 	}

 	randomizedIndices = randomizedIndices2;
 }

 MatrixXd getNonboundSubset(MatrixXd lagrangeMultipliers, double marginParameter)
 {
 	MatrixXd tempNonboundSubset = MatrixXd::Zero(lagrangeMultipliers.size(), 1); //n
 	int sampleCount = lagrangeMultipliers.size();
 	int subsetSize = 0;

 	for (int sampleNumber = 0; sampleNumber < sampleCount; sampleNumber++)
 	{
 		if ((lagrangeMultipliers(sampleNumber) != 0) && (lagrangeMultipliers(sampleNumber) != marginParameter))
 		{
 			tempNonboundSubset(subsetSize) = double(sampleNumber);
 			subsetSize = subsetSize + 1;
 		}
 	}

 	MatrixXd nonboundSubset = tempNonboundSubset.block(0, 0, subsetSize, 1);

 	return nonboundSubset;
 }

 void optimizePair(MatrixXd labelSet, MatrixXd gramMatrix, int pairFirst, int pairSecond, MatrixXd sampleError, MatrixXd &lagrangeMultipliers, double const marginParameter, double &bias, bool &flagOptimizeSuccess)
 {
	double const roundingError = 0.0000000000001;
 	double lowBoundary;
 	double highBoundary;
 	double secondDerivative;
 	double newSecondLagrangeMultiplier;
 	double newFirstLagrangeMultiplier;
 	double firstFunction;
 	double secondFunction;
 	double lowFunction;
 	double highFunction;
 	double lowObjectiveFunction;
 	double highObjectiveFunction;
 	double pairSecondBias;
 	double pairFirstBias;
 	double s = labelSet(pairFirst)*labelSet(pairSecond);
 	flagOptimizeSuccess = false;


 	sampleError(pairFirst) = ((lagrangeMultipliers.cwiseProduct(labelSet)).cwiseProduct(gramMatrix.col(pairFirst))).sum() - labelSet(pairFirst) - bias;
 	sampleError(pairSecond) = ((lagrangeMultipliers.cwiseProduct(labelSet)).cwiseProduct(gramMatrix.col(pairSecond))).sum() - labelSet(pairSecond) - bias;

 	if (pairFirst == pairSecond)
 	{
 		flagOptimizeSuccess = false;
 		return;
 	}

 	if (labelSet(pairSecond) == labelSet(pairFirst))
 	{
 		lowBoundary = getBiggerNumber(0.0, double(lagrangeMultipliers(pairFirst) + lagrangeMultipliers(pairSecond) - marginParameter));
 		highBoundary = getSmallerNumber(marginParameter, double(lagrangeMultipliers(pairFirst) + lagrangeMultipliers(pairSecond)));
 	}
 	else
 	{
 		lowBoundary = getBiggerNumber(0.0, double(lagrangeMultipliers(pairSecond) - lagrangeMultipliers(pairFirst)));
 		highBoundary = getSmallerNumber(marginParameter, double(marginParameter + lagrangeMultipliers(pairSecond) - lagrangeMultipliers(pairFirst)));
 	}

 	if (lowBoundary == highBoundary)
 	{
 		flagOptimizeSuccess = false;
 		return;
 	}

 	secondDerivative = gramMatrix(pairSecond, pairSecond) + gramMatrix(pairFirst, pairFirst) - 2 * gramMatrix(pairFirst, pairSecond);

 	if (secondDerivative > 0)
 	{
 		newSecondLagrangeMultiplier = lagrangeMultipliers(pairSecond) + (labelSet(pairSecond)*(sampleError(pairFirst) - sampleError(pairSecond)) / secondDerivative);

 		if (newSecondLagrangeMultiplier < lowBoundary)
 		{
 			newSecondLagrangeMultiplier = lowBoundary;
 		}
 		else
 		{
 			if (newSecondLagrangeMultiplier > highBoundary)
 			{
 				newSecondLagrangeMultiplier = highBoundary;
 			}
 		}
 	}
 	else
 	{
 		firstFunction = labelSet(pairFirst)*(sampleError(pairFirst) + bias) - lagrangeMultipliers(pairFirst)*gramMatrix(pairFirst, pairFirst) - s*lagrangeMultipliers(pairSecond)*gramMatrix(pairFirst, pairSecond);
 		secondFunction = labelSet(pairSecond)*(sampleError(pairSecond) + bias) - s*lagrangeMultipliers(pairFirst)*gramMatrix(pairFirst, pairSecond) - lagrangeMultipliers(pairSecond)*gramMatrix(pairSecond, pairSecond);
 		lowFunction = lagrangeMultipliers(pairFirst) + s*(lagrangeMultipliers(pairSecond) - lowBoundary);
 		highFunction = lagrangeMultipliers(pairFirst) + s*(lagrangeMultipliers(pairSecond) - highBoundary);

 		lowObjectiveFunction = lowFunction*firstFunction + lowBoundary*secondFunction + 0.5*(lowFunction*lowFunction)*gramMatrix(pairFirst, pairFirst) + 0.5*lowBoundary*lowBoundary*gramMatrix(pairSecond, pairSecond) + s*lowBoundary*lowFunction*gramMatrix(pairFirst, pairSecond);
 		highObjectiveFunction = highFunction*firstFunction + highBoundary*secondFunction + 0.5*(highFunction*highFunction)*gramMatrix(pairFirst, pairFirst) + 0.5*highBoundary*highBoundary*gramMatrix(pairSecond, pairSecond) + s*highBoundary*highFunction*gramMatrix(pairFirst, pairSecond);

 		if (lowObjectiveFunction < (highObjectiveFunction - roundingError))
 		{
 			newSecondLagrangeMultiplier = lowBoundary;
 		}
 		else
 		{
 			if (lowObjectiveFunction >(highObjectiveFunction + roundingError))
 			{
 				newSecondLagrangeMultiplier = highBoundary;
 			}
 			else
 			{
 				newSecondLagrangeMultiplier = lagrangeMultipliers(pairSecond);
 			}
 		}
 	}

 	if (abs(newSecondLagrangeMultiplier - lagrangeMultipliers(pairSecond)) < roundingError*(newSecondLagrangeMultiplier + lagrangeMultipliers(pairSecond) + roundingError))
 	{
 		flagOptimizeSuccess = false;
 		return;
 	}

 	newFirstLagrangeMultiplier = lagrangeMultipliers(pairFirst) + s*(lagrangeMultipliers(pairSecond) - newSecondLagrangeMultiplier);

 	pairSecondBias = sampleError(pairSecond) + labelSet(pairSecond) * (newFirstLagrangeMultiplier - lagrangeMultipliers(pairFirst))*gramMatrix(pairFirst, pairSecond)
 		+ labelSet(pairSecond)*(newSecondLagrangeMultiplier - lagrangeMultipliers(pairSecond))*gramMatrix(pairSecond, pairSecond) + bias;
 	pairFirstBias = sampleError(pairFirst) + labelSet(pairSecond) * (newFirstLagrangeMultiplier - lagrangeMultipliers(pairFirst))*gramMatrix(pairFirst, pairSecond)
 		+ labelSet(pairSecond)*(newSecondLagrangeMultiplier - lagrangeMultipliers(pairSecond))*gramMatrix(pairFirst, pairSecond) + bias;

 	if (newSecondLagrangeMultiplier < roundingError)
 	{
 		newSecondLagrangeMultiplier = 0.0;
 	}

 	if (newFirstLagrangeMultiplier < roundingError)
 	{
 		newFirstLagrangeMultiplier = 0.0;
 	}

 	lagrangeMultipliers(pairFirst) = newFirstLagrangeMultiplier;
 	lagrangeMultipliers(pairSecond) = newSecondLagrangeMultiplier;

 	if (0 < lagrangeMultipliers(pairSecond) && lagrangeMultipliers(pairSecond) < marginParameter)
 	{
 		bias = pairSecondBias;
 	}
 	else
 	{
 		if (0 < lagrangeMultipliers(pairFirst) && lagrangeMultipliers(pairFirst) <marginParameter)
 		{
 			bias = pairFirstBias;
 		}
 		else
 		{
 			bias = (pairSecondBias + pairFirstBias) / 2;
 		}
 	}

 	flagOptimizeSuccess = true;
 }

 void examineSample(int pairSecond, MatrixXd &sampleError, MatrixXd &lagrangeMultipliers, MatrixXd gramMatrix, MatrixXd labelSet, double &bias, double const marginParameter, double const tolerance, bool &flagExamineSuccess)
 {
 	flagExamineSuccess = false;
 	MatrixXd randomizedIndices;
 	MatrixXd nonboundSubset;
 	bool flagOptimizeSuccess;
 	int pairFirst = 0;
 	VectorXd incrementFunction;


 	sampleError(pairSecond) = ((lagrangeMultipliers.cwiseProduct(labelSet)).cwiseProduct(gramMatrix.col(pairSecond))).sum() - bias - labelSet(pairSecond);

 	if ((labelSet(pairSecond)*sampleError(pairSecond) < -tolerance && lagrangeMultipliers(pairSecond) < marginParameter) ||
 		(labelSet(pairSecond)*sampleError(pairSecond) > tolerance && lagrangeMultipliers(pairSecond) > 0))
 	{
 		nonboundSubset = getNonboundSubset(lagrangeMultipliers, marginParameter);

 		if (nonboundSubset.size() > 1)
 		{
 			double maximalError = 0.000;
 			int sampleCount = lagrangeMultipliers.size();

 			for (int evaluatedSample = 0; evaluatedSample < sampleCount; evaluatedSample++)
 			{
 				if (pairSecond != evaluatedSample)
 				{
 					sampleError(evaluatedSample) = ((lagrangeMultipliers.cwiseProduct(labelSet)).cwiseProduct(gramMatrix.col(evaluatedSample))).sum() - bias - labelSet(evaluatedSample);
 					if (abs(sampleError(pairSecond) - sampleError(evaluatedSample)) > maximalError)
 					{
 						maximalError = abs(sampleError(pairSecond) - sampleError(evaluatedSample));
 						pairFirst = evaluatedSample;
 					}
 				}
 			}

 			optimizePair(labelSet, gramMatrix, pairFirst, pairSecond, sampleError, lagrangeMultipliers, marginParameter, bias, flagOptimizeSuccess);

 			if (flagOptimizeSuccess == true)
 			{
 				flagExamineSuccess = true;
 				return;
 			}
 			else
 			{
 				flagExamineSuccess = false;
 			}


 			randomizeIndexOrder(nonboundSubset, randomizedIndices);

 			for (int sampleNumber = 0; sampleNumber < randomizedIndices.size(); sampleNumber++)
 			{
 				pairFirst = int(randomizedIndices(sampleNumber));

 				optimizePair(labelSet, gramMatrix, pairFirst, pairSecond, sampleError, lagrangeMultipliers, marginParameter, bias, flagOptimizeSuccess);

 				if (flagOptimizeSuccess == true)
 				{
 					flagExamineSuccess = true;
 					return;
 				}
 				else
 				{
 					flagExamineSuccess = false;
 				}
 			}
 		}

 		incrementFunction = VectorXd::LinSpaced(lagrangeMultipliers.size(), 0, lagrangeMultipliers.size() - 1);
 		randomizeIndexOrder(incrementFunction.matrix(), randomizedIndices);

 		for (int sampleNumber = 0; sampleNumber < randomizedIndices.size(); sampleNumber++)
 		{
 			pairFirst = int(randomizedIndices(sampleNumber));

 			optimizePair(labelSet, gramMatrix, pairFirst, pairSecond, sampleError, lagrangeMultipliers, marginParameter, bias, flagOptimizeSuccess);

 			if (flagOptimizeSuccess == true)
 			{
 				flagExamineSuccess = true;
 				return;
 			}
 			else
 			{
 				flagExamineSuccess = false;
 			}
 		}
 	}
 }

 void smosvm(MatrixXd trainSet, MatrixXd labelSet, double const marginParameter, MatrixXd &w, double &bias, float &trainingTime)
 {
 	//This function finds optimal hyperplane separating the values in trainSet
 	//based on labelSet and marginParameter value. To do so it uses Sequential Minimal Optimization

 	int sampleCount = labelSet.size();
 	double const tolerance = 0.001;
 	MatrixXd lagrangeMultipliers = MatrixXd::Zero(sampleCount, 1);
 	MatrixXd sampleError = MatrixXd::Zero(sampleCount, 1);
 	MatrixXd nonboundSubset;
 	MatrixXd const gramMatrix = trainSet*(trainSet.transpose()); //defines linear SVM
 	clock_t startTime = clock();

 	bool flagExamineAll = true;
 	bool flagHasChanged = false;
 	bool flagExamineSuccess = false;

 	while (flagExamineAll == true || flagHasChanged == true)
 	{
 		flagHasChanged = false;

 		if (flagExamineAll == true)
 		{
 			for (int pairSecond = 0; pairSecond < sampleCount; pairSecond++)
 			{
 				examineSample(pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter, tolerance, flagExamineSuccess);

 				if (flagExamineSuccess == true)
 				{
 					flagHasChanged = true;
 				}
 			}
 		}
 		else
 		{
 			nonboundSubset = getNonboundSubset(lagrangeMultipliers, marginParameter);

 			for (int sampleNumber = 0; sampleNumber < nonboundSubset.size(); sampleNumber++)
 			{
 				int pairSecond = int(nonboundSubset(sampleNumber));

 				examineSample(pairSecond, sampleError, lagrangeMultipliers, gramMatrix, labelSet, bias, marginParameter, tolerance, flagExamineSuccess);

 				if (flagExamineSuccess == true)
 				{
 					flagHasChanged = true;
 				}
 			}
 		}

 		if (flagExamineAll == true)
 		{
 			flagExamineAll = false;
 		}
 		else
 		{
 			if (flagHasChanged == false)
 			{
 				flagExamineAll = true;
 			}
 		}
 	}

 	w = (((lagrangeMultipliers.cwiseProduct(labelSet)).transpose())*trainSet).transpose();
 	clock_t endTime = clock();
 	trainingTime = (float(endTime - startTime)) / CLOCKS_PER_SEC;
 }


 bool loadData(std::string filename, MatrixXd &dataSet, MatrixXd &labelSet)
{
	std::fstream dataFile;
	dataFile.open(filename);
	std::string line;
	bool loadSuccess = false;

	if (dataFile.is_open())
	{
		int sampleCount = 0;
		int featureCount = 0;

		while (!dataFile.eof())
		{
			getline(dataFile, line);
			if (line != "")
			{
				sampleCount++;
			}
		}

		dataFile.clear();
		dataFile.seekg(0, std::ios::beg);

		getline(dataFile, line);
		for (int i = 0; i < int(line.length()); i++)
		{
			if (line[i] == ',')
			{
				featureCount++;
			}
		}

		dataSet = MatrixXd::Zero(sampleCount, featureCount);
		labelSet = MatrixXd::Zero(sampleCount, 1);
		std::string featureValue;
		int rowNumber = 0;
		int featureNumber;

		dataFile.clear();
		dataFile.seekg(0, std::ios::beg);

		while (!dataFile.eof())
		{
			getline(dataFile, line);

			if (line == "")
			{
				break;
			}
			featureValue = "";
			featureNumber = 0;

			for (int i = 0; i <= int(line.length()); i++)
			{
				if (i == int(line.length()))
				{
					labelSet(rowNumber, 0) = std::stod(featureValue);
				}
				if (line[i] == ',')
				{
					dataSet(rowNumber, featureNumber) = std::stod(featureValue);
					featureValue = "";
					featureNumber++;
					continue;
				}
				featureValue += line[i];
			}
			rowNumber++;
		}
		loadSuccess = true;
		std::cout << "\n Data loaded successfully!" << std::endl;
		return loadSuccess;
	}
	else
	{
		std::cout << "\n Something went wrong while opening the file, please make sure the files are in the same directory as the program and try again." << std::endl;
		return loadSuccess;
	}
}

MatrixXd svmclassify(MatrixXd w, double bias, MatrixXd &dataSet, float &classificationTime)
{
	clock_t startTime = clock();

	int sampleCount = dataSet.rows();
	MatrixXd resultSet = MatrixXd::Zero(sampleCount, 1);
	MatrixXd supportVectors = w;

	for (int sampleNumber = 0; sampleNumber < sampleCount; sampleNumber++)
	{
		if (((supportVectors.cwiseProduct(dataSet.row(sampleNumber).transpose())).sum() - bias) > 0)
		{
			resultSet(sampleNumber, 0) = 1;
		}
		else
		{
			resultSet(sampleNumber, 0) = -1;
		}
	}

	clock_t endTime = clock();
	classificationTime = (float(endTime - startTime)) / CLOCKS_PER_SEC;

	return resultSet;
}

float checkAccuracy(MatrixXd const resultSet, MatrixXd const &trueResultSet)
{
	float accuracy = 0;
	int properlyClassifiedCount = 0;

	for (int sampleNumber = 0; sampleNumber < resultSet.rows(); sampleNumber++)
	{
		if (resultSet(sampleNumber, 0) == trueResultSet(sampleNumber, 0))
		{
			properlyClassifiedCount++;
		}
	}

	accuracy = (float(properlyClassifiedCount) / float(resultSet.rows()))*100;
	return accuracy;
}

void saveResult(float const &trainingTime, float const &classificationTime, float const &accuracy)
{
	std::fstream resultsFile;
	resultsFile.open("linear SVM results.txt", std::ios::app);

	if (resultsFile.is_open())
	{
		resultsFile << "---------------" << std::endl;
		resultsFile << "Time taken to train the SVM: " << trainingTime << " seconds." << std::endl;
		resultsFile << "Time taken to classify: " << classificationTime << " seconds." << std::endl;
		resultsFile << "Classification accuracy: " << accuracy << "% ." << std::endl;

		resultsFile.close();
		return;
	}
	else
	{
		std::cout << "Error: file couldn't be opened" << std::endl;
		return;
	}
}

 int main()
{
	bool flagExit = false;
	bool flagTrainingSetExists = false;
	bool loadSuccess = false;
	char userInput;
	std::string filename;
	enum menuStates { waitForUserInput, trainSVM, useSVM, quit };
	enum menuStates currentState = waitForUserInput;
	MatrixXd trainSet;
	MatrixXd labelSet;
	MatrixXd testSet;
	MatrixXd trueResultSet;
	MatrixXd resultSet;
	float classificationTime = 0;
    double marginParameter = 0.001;
 	double bias = 0.0;
 	MatrixXd w;
 	float trainingTime;
 	float accuracy;

	std::cout << "Linear SVM Heartbeat Classification" << std::endl;

	while (flagExit == false)
	{
		switch (currentState)
		{
		case waitForUserInput:
			std::cout << "\n Please select desired operation by pressing appropriate key: \n 1.(T)rain SVM \n 2.(C)lassify signal using trained SVM \n 3.(Q)uit" << std::endl;
			std::cin >> userInput;
			if (userInput == 'Q' || userInput == 'q')
			{
				currentState = quit;
			}
			else if (userInput == 'T' || userInput == 't')
			{
				currentState = trainSVM;
			}
			else if (userInput == 'C' || userInput == 'c')
			{
				currentState = useSVM;
			}
			else
			{
				std::cout << "\n Unrecognized input, please try again." << std::endl;
			}
			break;

		case trainSVM:

			std::cout << "\n Loading training data." << std::endl;

			loadSuccess = false;
			std::cout << "\n Please enter the training dataset filename." << std::endl;
			std::cin >> filename;
			loadSuccess = loadData(filename, trainSet, labelSet);

			if (loadSuccess == false)
			{
				currentState = waitForUserInput;
				break;
			}

			std::cout << "\n Training in progress, please wait." << std::endl;
			smosvm(trainSet, labelSet, 0.001, w , bias, trainingTime);

			std::cout << "\n Training finished!" << std::endl;
			flagTrainingSetExists = true;
			currentState = waitForUserInput;
			break;

		case useSVM:
			if (flagTrainingSetExists == false)
			{
				std::cout << "\n SVM has not been trained, please train it first" << std::endl;
				currentState = waitForUserInput;
				break;
			}

			loadSuccess = false;
			std::cout << "\n Please enter the testing dataset filename." << std::endl;
			std::cin >> filename;
			loadSuccess = loadData(filename, testSet, trueResultSet);

			if (loadSuccess == false)
			{
				currentState = waitForUserInput;
				break;
			}

			resultSet = svmclassify(w, bias, testSet, classificationTime);
			accuracy = checkAccuracy(resultSet, trueResultSet);

			saveResult(trainingTime, classificationTime, accuracy);
			std::cout << "\n Classification results have been saved in >linear SVM results.txt< file." << std::endl;
			currentState = waitForUserInput;
			break;

		case quit:
			std::cout << "\n Press ENTER key to quit." << std::endl;
			std::cin.get();
			std::cin.ignore();
			flagExit = 1;
			break;

		default:
			std::cout << "Default case, something went really wrong." << std::endl;
			currentState = waitForUserInput;
			break;
		}
	}

	return 0;
}
