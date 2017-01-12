#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <time.h>


using namespace std;


int FileLength(const char* filename){

    int lines = 0;
    std::string line;
    std::ifstream myfile(filename);
    
    while (std::getline(myfile, line)){
        ++lines;
    }
      

    return lines;

}



Eigen::MatrixXd readCSV(const char* file, int rows, int cols) {
 // kod źródłowy: https://gist.github.com/infusion/43bd2aa421790d5b4582#file-read-csv-to-eigen-cpp   
  std::ifstream in(file);
  
  std::string line;

  int row = 0;
  int col = 0;

  Eigen::MatrixXd data = Eigen::MatrixXd(rows, cols);

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          data(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      data(row, col) = atof(start);

      row++;
    }

    in.close();
  }
  return data;
}



Eigen::MatrixXd LDA_model(const Eigen::MatrixXd& training1, const Eigen::MatrixXd& training2){
    
    
    //liczba cech
    int featureNumber = training1.cols();
    // liczba probek klasy 1 zbioru uczacego
    double c1 = training1.rows();
    // liczba probek klasy 2 zbioru uczacego
    double c2 = training2.rows();
    // liczba wszystkich probek zbioru uczacego
    double N;
    N = c1 + c2;
    // prawdopodobienstwo wystapienia klasy 1
    double piC1;
    piC1 = c1/N;
    // prawdopodobienstwo wystapienia klasy 2
    double piC2;
    piC2 = c2/N;
    
    
    
    // mu
    Eigen::RowVectorXd u1(featureNumber); //dekalaracja wektora mu dla klasy 1
    u1 = (training1.colwise().sum())/c1;
    Eigen::RowVectorXd u2(featureNumber); //deklaracja wektora mu dla klasy 2
    u2 = (training2.colwise().sum())/c2;
    
    //sigma
    
    Eigen::MatrixXd E1(featureNumber, featureNumber);
    Eigen::MatrixXd E2(featureNumber, featureNumber);
    Eigen::MatrixXd sigma(featureNumber, featureNumber);
    Eigen::MatrixXd t(featureNumber, featureNumber);
    
    for(int i=0; i<c1; i++){
        t= ((training1.row(i)-u1).transpose()*(training1.row(i)-u1))/(N-2);
        E1 = E1 + t;
    }
    for(int i=0; i<c2; i++){
        t= ((training2.row(i)-u2).transpose()*(training2.row(i)-u2))/(N-2);
        E2 = E2 + t;
    }
    
    sigma = E1+E2;
    
    
    // obliczenie wspolczynnikow funkcji klasyfikujacej
    Eigen::RowVectorXd u(featureNumber);
    Eigen::MatrixXd a(featureNumber,1);
    Eigen::MatrixXd a0(1,1);
    Eigen::MatrixXd logpi(1,1);
    
    u = u1-u2;
    a = sigma.inverse()*u.transpose();
    logpi << log(piC1/piC2);
    a0 = logpi-(((u1+u2)*sigma.inverse()*(u1-u2).transpose())/2);
    


    Eigen::MatrixXd modela((featureNumber+1),1);


    for (int i=0;i<featureNumber;i++){
        modela(i,0)=a(i,0);
        }

    modela(featureNumber,0)=a0(0,0);

    
    return modela;
  
    
}


Eigen::VectorXd LDA_classify(const Eigen::MatrixXd& test, const Eigen::MatrixXd& model){
    
    int featureNumber = test.cols();    

    // Dlugosc zbioru testowego
    int testLen = test.rows();
    // wektor wynikow
    Eigen::VectorXd klasyfikacja(testLen);
    Eigen::MatrixXd f(1,1);
    Eigen::MatrixXd temp(1,1);
    temp(0,0)=0;
    
   
   for(int i=0; i<testLen; i++){
        f << 0;
        for(int j=0; j<featureNumber; j++){
            temp << test(i,j)*model(j,0);
            
            f = f+temp;
            
        }

        Eigen::MatrixXd a0(1,1);

        a0(0,0)=model(featureNumber,0);

        f = f + a0;
        if(f(0,0)<0){
            klasyfikacja(i)=-1;
        }
        else{
            klasyfikacja(i)=1;
        }
    }


    return klasyfikacja;
}


int main(){

    int featureNumber=5;
    const char*  VEtrain="training1.csv";
   const char*  Ntrain="training2.csv";
   const char* VENtest="test.csv";

    int VElength=FileLength(VEtrain);
    int Nlength=FileLength(Ntrain);
    int testlength=FileLength(VENtest);

    Eigen::MatrixXd VEtr = readCSV(VEtrain, VElength, featureNumber);
    Eigen::MatrixXd Ntr = readCSV(Ntrain, Nlength, featureNumber);
    Eigen::MatrixXd test = readCSV(VENtest, testlength, featureNumber);

    clock_t startTrain = clock();

    Eigen::MatrixXd model = LDA_model(VEtr, Ntr);

    clock_t endTrain = clock();
    double trainingTime = (double) (endTrain-startTrain) / CLOCKS_PER_SEC * 1000.0;


    clock_t startClassify = clock();

    Eigen::VectorXd result = LDA_classify(test, model);
    
    clock_t endClassify = clock();
    double ClassTime = (double) (endClassify-startClassify) / CLOCKS_PER_SEC * 1000.0;
    
    cout<<"Czas uczenia: "<<trainingTime<<endl;
    cout<<"Czas klasyfikacji: "<<ClassTime<<endl;
    
    ofstream outfile ("classify_result.csv");

    if (outfile.is_open()){
    
        for(int i=0;i<testlength;i++){
            outfile<<result(i)<<";";
        }
    }

    outfile.close();
    
}