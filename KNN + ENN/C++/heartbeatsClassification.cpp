#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Data
{
private:
	MatrixXf matrix;
	MatrixXf labels;
public:
	Data(string pathToData, string pathToLabels, int numberOfRows, int numberOfColumns);
	Data(string pathToData, int numberOfRows, int numberOfColumns);

	MatrixXf getMatrix(){return (matrix);};
	MatrixXf getLabels(){return (labels);};
};

MatrixXf doKNNClasificationForAllTestSamples(Data testData, Data trainData, int k);
VectorXf doKNNClasification(VectorXf testDataVector, Data trainData, int k);

MatrixXf doENNClasificationForAllTestSamples(Data testData, Data trainData);
int doENNClasification(VectorXf testDataVector, Data trainData);

MatrixXf ReadCSV(string filename, int number_of_features, char delimiter);
int calculateMode(VectorXf vector);
float calculateEffectiveness(MatrixXf results, MatrixXf labels);

int main() {

	Data trainData("trainset.csv", 300, 6);
	Data testData("testset.csv", 100, 6);

	/* calculate KNN */
	int k = 5;
	MatrixXf knnResults = doKNNClasificationForAllTestSamples(testData, trainData, k);

	/* calculate ENN */
	MatrixXf ennResults = doENNClasificationForAllTestSamples(testData, trainData);

	cout << "Skutecznosc KNN: " << calculateEffectiveness(knnResults, testData.getLabels()) << "%" << endl;
	cout << "Skutecznosc ENN: " << calculateEffectiveness(ennResults, testData.getLabels()) << "%" << endl;

	return 0;
}

Data::Data(string pathToData,int numberOfRows, int numberOfColumns) {
	matrix = MatrixXf::Zero(numberOfRows,numberOfColumns);
	labels = MatrixXf::Zero(numberOfRows,1);

	matrix = ReadCSV(pathToData, numberOfColumns-1,',');
	labels = matrix.rightCols(1);
	matrix = matrix.leftCols(numberOfColumns-1);
}

MatrixXf doKNNClasificationForAllTestSamples(Data testData, Data trainData, int k) {
	MatrixXf resultMatrix(testData.getMatrix().rows(), 1);
	for (int testSampleRow = 0; testSampleRow < testData.getMatrix().rows(); testSampleRow++) {
		VectorXf currentRow = testData.getMatrix().row(testSampleRow);
		VectorXf predictedClasses = doKNNClasification(currentRow, trainData, k);
		resultMatrix(testSampleRow, 0) = calculateMode(predictedClasses);
	}
	return resultMatrix;
}

VectorXf doKNNClasification(VectorXf testDataVector, Data trainData, int k) {
	MatrixXf trainDataMatrix = trainData.getMatrix();
	VectorXf euclideanDistance(trainDataMatrix.rows());
	//every column represents distance between test sample and train samples
	for (int trainDataRow = 0; trainDataRow < trainDataMatrix.rows(); trainDataRow++) {
		float distance = 0.0;
		for (int column = 0; column < trainDataMatrix.cols(); column++) {
			distance = distance + pow(testDataVector(column) - trainDataMatrix(trainDataRow,column),2.0);
		}
		euclideanDistance(trainDataRow) = sqrt(distance);
	}
	//looking for k-nearest neighbors for test sample
	VectorXf predictedClass(k);
	for (int neighbour = 0; neighbour < k; neighbour++) {
		MatrixXf trainDataLabels = trainData.getLabels();
		VectorXf::Index minIndex;
		euclideanDistance.minCoeff(&minIndex);
		euclideanDistance(minIndex) = testDataVector.maxCoeff();
		predictedClass(neighbour) = trainDataLabels(minIndex);
	}
	return predictedClass;
}

MatrixXf doENNClasificationForAllTestSamples(Data testData, Data trainData) {
	MatrixXf resultMatrix(testData.getMatrix().rows(), 1);
	for (int testSampleRow = 0; testSampleRow < testData.getMatrix().rows(); testSampleRow++) {
		VectorXf currentRow = testData.getMatrix().row(testSampleRow);
		int predictedClass = doENNClasification(currentRow, trainData);
		resultMatrix(testSampleRow, 0) = predictedClass;
	}
	return resultMatrix;
}

int doENNClasification(VectorXf testDataVector, Data trainData) {
	int cnt=0;
	int numberOfIteration = round(sqrt(trainData.getMatrix().rows()));
	VectorXf results(numberOfIteration/2);
	for (int k=1; k <= numberOfIteration; k+=2) {
		VectorXf predictedClasses = doKNNClasification(testDataVector, trainData, k);
		map<int, float> predictedClassesMap;
		map<int, float>::iterator it;
		for (int i=0; i < predictedClasses.size(); i++) {
			float weight = 1/(log2(i+1));
			int predictedClass = predictedClasses(i);
			it = predictedClassesMap.find(predictedClasses(i));
			if (it != predictedClassesMap.end())
				it->second += weight;
			else
				predictedClassesMap.insert(std::pair<int,float>(predictedClass, weight));
		}
		int returnClass;
		float returnWeight = 0;
		for (auto& iterator: predictedClassesMap) {
			if (iterator.second > returnWeight) {
				returnClass = iterator.first;
			}
		}
		results(cnt) = returnClass;
		cnt++;
	}
	return calculateMode(results);
}

int calculateMode(VectorXf vector) {
	int mode, max, cnt;
	sort(vector.data(), vector.data()+vector.size());
	mode = vector(0);
	for (int i = 0; i < vector.size(); i++) {
		if (vector(i) == mode)
			cnt++;
		else
			cnt = 0;

		if ( cnt > max ) {
			max = cnt;
			mode = vector(i);
		};
	}
	return (mode);
}

float calculateEffectiveness(MatrixXf results, MatrixXf labels) {
	int counter = 0;
	for (int row = 0; row < results.rows(); row++) {
		if (labels(row) == results(row)) {
			counter++;
		}
	}
	return (counter*100/results.rows());
}

MatrixXf ReadCSV(string filename, int number_of_features, char delimiter) {
    ifstream file (filename.c_str());
    MatrixXf inputdata(0,0);
    string value;
    int i =0, j =0;

    while(getline (file, value)) {
        j = 0;
        stringstream iss(value);
        string result;
        double a;
        inputdata.conservativeResize (inputdata.rows()+1,number_of_features + 1 );
        while( getline(iss,result,delimiter)) {
            a = strtod(result.c_str(),NULL);
            inputdata(i,j) = a;
            j = j + 1;
        }
        i = i + 1;
    }
    file.close();
    return inputdata;
}


