#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <Eigen/Dense>
#include <map>
#include <windows.h>
#include <time.h>

using namespace std;
using Eigen::MatrixXf;
using Eigen::Matrix;
using Eigen::VectorXi;
typedef std::map<int, int> BasePairMap;


void StartCounter(double &PCFreq,__int64 &CounterStart)
{
    LARGE_INTEGER Largeint;
    QueryPerformanceFrequency(&Largeint);
    PCFreq = double(Largeint.QuadPart)/1000.0f;

    QueryPerformanceCounter(&Largeint);
    CounterStart = Largeint.QuadPart;

}
double GetCounter(double PCFreq, __int64 CounterStart)
{
    LARGE_INTEGER Largeint;
    QueryPerformanceCounter(&Largeint);
    return (double(Largeint.QuadPart - CounterStart)/PCFreq);

}

MatrixXf ReadCSV(string filename, int number_of_features, char delimiter)
{
    ifstream file (filename.c_str());
    MatrixXf inputdata(0,0);
    string value;
    int i =0, j =0;

    while(getline ( file, value))
    {
        j = 0;
        stringstream iss(value);
        string result;
        double a;
        inputdata.conservativeResize (inputdata.rows()+1,number_of_features + 1 );
        while( getline(iss,result,delimiter))
        {
            a = strtod(result.c_str(),NULL);
            inputdata(i,j) = a;
            j = j + 1;
        }
        i = i + 1;
    }

    file.close();

    return inputdata;
}



int main()
{
    // Measuring time
    double PerformaceCounter = 0.0f;
    __int64 CounterStart= 0;

    int number_of_classes = 2;
    int number_of_features  = 5;
    int number_of_features_to_classification;
    int i =0, j =0;

    //Map of class
    BasePairMap map_class;
    map_class[0] = 1;
    map_class[1] = -1;

    //Reading training data
    string filename = "trainset.csv";
//    cout<< "Enter the train file: ";
//    cin>>filename;


//    cout<< "Enter the number of features in dataset: ";
//    cin>>number_of_features;

    cout<< "Enter the number of features which will be used in classifier (1-5):";
    cin>> number_of_features_to_classification;
    MatrixXf inputdata = ReadCSV(filename,number_of_features,',');

   StartCounter(PerformaceCounter,CounterStart);
//    cout<< "Enter the test file: ";
//    cin>> filename;
    //Division for class
    MatrixXf input_1(0, 0); // class 1
    MatrixXf input_2(0, 0); //class 2

    for(i = 0;i<inputdata.rows();i++)
    {
        if (inputdata(i,number_of_features)== map_class[0])
        {
            input_1.conservativeResize (input_1.rows() + 1,number_of_features+1);
            input_1.row(input_1.rows()-1) = inputdata.row(i);
        }
        else if(inputdata(i,number_of_features)== map_class[1])
        {
            input_2.conservativeResize (input_2.rows() + 1,number_of_features+1);
            input_2.row(input_2.rows()-1) = inputdata.row(i);
        }
    }

    //Incidence of class
    MatrixXf fy(1,number_of_classes);
    fy(0,0) = input_1.rows()/double(inputdata.rows());
    fy(0,1) = input_2.rows()/double(inputdata.rows());

    //Calculation of average value of each features for each class
    MatrixXf mean_class(2,number_of_features_to_classification);
    mean_class.row(0) = input_1.leftCols(number_of_features_to_classification).array().colwise().mean();
    mean_class.row(1) = input_2.leftCols(number_of_features_to_classification).array().colwise().mean();

    //Calculation of standard deviation for each class
    MatrixXf std_class(2,number_of_features_to_classification);

    std_class.row(0) = ((input_1.rowwise() - input_1.colwise().mean()).leftCols(number_of_features_to_classification).array().pow(2).colwise().sum()/(input_1.rows()-1)).sqrt();
    std_class.row(1) = ((input_2.rowwise() - input_2.colwise().mean()).leftCols(number_of_features_to_classification).array().pow(2).colwise().sum()/(input_2.rows()-1)).sqrt();

    ofstream results("results.txt");
    results<<"Learning time: " << GetCounter(PerformaceCounter,CounterStart)<<"ms\n";
    //Learning time
    cout<<"Learning time: " <<GetCounter(PerformaceCounter,CounterStart)<<"ms"<<endl;

    //Reading test data set
    filename = "testset.csv";

    MatrixXf input_test =ReadCSV(filename,number_of_features,',');


    //Calculation of  probability density function for every data in test set
    double constant; //constant factor in probability function
    constant = 1/sqrt(2*M_PI);
    MatrixXf probability(input_test.rows(),number_of_classes); //class probability for each sample in test set
    MatrixXf temp(number_of_classes,number_of_features_to_classification);

    for(i=0;i<input_test.rows();i++)
   {
       for(j=0;j<number_of_classes;j++)
       {
           temp.row(j) = input_test.leftCols(number_of_features_to_classification).row(i);
       }
        temp = -(temp - mean_class).array().pow(2)/(2*std_class.array().pow(2));
        temp = temp.array().exp()*std_class.array().pow(-1)*constant;
        probability.row(i) = temp.rowwise().prod().transpose().cwiseProduct(fy);

    }

    MatrixXf::Index   maxIndex[2];
    VectorXi maxVal(probability.rows());

   for(i=0;i<probability.rows();i++)
   {
       probability.row(i).maxCoeff( &maxIndex[1] );

       maxVal(i) = maxIndex[1];
   }

    //Checking data form classifier with real data
    int good = 0;
    int notgood = 0;

    MatrixXf statistic(number_of_classes,3);
    // column 1 - number of correct classified data
    // column 2 - number of incorrect classified data
    // column 3 - ration correct/(correct + incorrect)

    for (i=0;i<number_of_classes;i++)
    {
        for(j=0;j<input_test.rows();j++)
        {
            if(maxVal(j) == i && map_class[maxVal(j)] == input_test(j,number_of_features))
            {
//                if(map_class[maxVal(j)] == input_test(j,5))
//                {
                    good++;
            }
            else if(input_test(j,number_of_features) == map_class[i])
                {
                    notgood++;
                }
        }

        statistic(i,0) = good;
        statistic(i,1) = notgood;
        statistic(i,2) = double(good)/(notgood+good);
        good = 0;
        notgood = 0;
    }
    cout<< statistic<<endl;
    ofstream klasyf("classification.txt");
    klasyf<<maxVal;
    results<<statistic<<"\n";
    good = statistic.leftCols(1).sum();

    //Effectiveness of classifier - ratio: good classified  to all data
    double effective;
    effective = double(good)/input_test.rows()*100;
    results<<"Effectiveness: "<<effective << "%\n";
    cout<<"Effectiveness: "<<effective<<" %"<<endl;
    cout<<"Execution time: " <<GetCounter(PerformaceCounter,CounterStart)<<"ms"<<endl;
    results<<"Execution time: "<<GetCounter(PerformaceCounter,CounterStart)<<"ms\n";
    return 0;
}
