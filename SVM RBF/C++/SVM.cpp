#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <time.h>

#include <Eigen/Dense>
using namespace Eigen;

struct SVMOptions {
	/*
	struktura zawierająca parametry uczenia algorytmu; konstruktor domyślny 
	ustawia domyślne dla tego algorytmu wartości
	*/
   SVMOptions(): C(1.0), tol(0.0001), sigma(0.5), iterLimit(10000), passLimit(10) { }   // default Constructor
   double C, tol, sigma;
   unsigned int iterLimit, passLimit;
};

struct ParsedData {
	/*
	typ danych wyjściowych funkcji parseData, zawiera zbiory: treningowy, uczący 
	oraz wektory celu dla obu tych zbiorów
	*/
	MatrixXd trainSet, testSet;
	VectorXd trainSetOut, testSetOut;
};

template<typename M>
M load_csv (const std::string & path) {
	/*
	wczytywanie danych o nieznanej ilości wierszów i kolumn z pliku CSV
	*/
    std::ifstream indata;
    indata.open(path.c_str());
    std::string line;
    double value;
    std::vector<double> values;
    unsigned int rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
             values.push_back(strtod(cell.c_str(), NULL));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

VectorXd extractLabelsVec(MatrixXd& matrixFromCSV){
	/*
	modyfikacja macierzy wejściowej danych (usunięcie ostatniej kolumny zawierającej
	 poprawne przyporządkowanie) oraz utworzenie dodatkowego wektora zawierającego 
	 przyporządkowanie
	*/
	VectorXd tmp = matrixFromCSV.rightCols(1);
	unsigned int numRows = matrixFromCSV.rows();
	unsigned int numCols = matrixFromCSV.cols()-1;
	matrixFromCSV.conservativeResize(numRows, numCols);
	return tmp;
}

class SVM{
	private:
		double C;
		double tol;
		double sigma;
		unsigned int iterLimit;
		unsigned int passLimit;
		MatrixXd data;
		VectorXd labels;
		VectorXd alphas;
		unsigned int n;
		unsigned int d;
		double bias;
		
		double RBFKernel(const VectorXd&, const VectorXd&);
		double giveMargin(const VectorXd&);
		int randomWithExceptionOf(int, int, int);
	public:
		double trainTime;
		double testTime;

		SVM(const MatrixXd&, const VectorXd, SVMOptions);
		void train();
		int predict(const VectorXd&);
		VectorXd massPredict(const MatrixXd&);
		VectorXd testModel(const MatrixXd&, const VectorXd&);
};

void SVM::train(){
	/*
	uczenie algorytmu SVM RBF
	*/

	clock_t startTrain = clock();

	unsigned int iter = 0;
	unsigned int passes = 0;

	unsigned int j, alphaFlag;
	double Ei, Ej, ai, aj, L, H, eta, newai, newaj, b1, b2;	

	
	while(passes < this->passLimit && iter < this->iterLimit){
		//WYKONUJ ALGORYTM SMO CHYBA ŻE LICZBA "PUSTYCH" PRZEBIEGÓW PRZEKROCZY
		//passLimit LUB ILOŚĆ ITERACJI PRZEKROCZY MAKSYMALNĄ ILOŚĆ ITERACJI
		alphaFlag = 0;
		for(unsigned int i = 0; i < this->n; i++){
			Ei = this->giveMargin(this->data.row(i)) - this->labels(i);
			if(((this->labels(i) * Ei) < (-tol) && this->alphas(i) < C) || ((this->labels(i) * Ei) > tol && this->alphas(i) > 0)){
				//WARTOŚCI WEKTORA ALPHAS(i) BĘDĄ AKTUALIZOWANE W TYM PRZEBIEGU PĘTLI
				j = this->randomWithExceptionOf(0, n-1, i);
				Ej = this->giveMargin(this->data.row(j)) - this->labels(j);
				
				ai = this->alphas(i);
				aj = this->alphas(j);
				L = 0; H = this->C;
				if(this->labels(i) == this->labels(j)){
					L = fmax(0, (ai+aj-H));
					H = fmin(H, (ai+aj));
				}else{
					L = fmax(0, (aj-ai));
					H = fmin(H, (H-ai+aj));
				}
				if(abs(L - H) < 0.0001) continue;
			
				eta = 2 * RBFKernel(this->data.row(i), this->data.row(j)) - RBFKernel(this->data.row(i), this->data.row(i)) - RBFKernel(this->data.row(j), this->data.row(j));
				if(eta >= 0) continue; 
				
				newaj = aj - (this->labels(j) * (Ei - Ej) / eta);
				if(newaj > H) newaj = H;
				if(newaj < L) newaj = L;
				if(abs(aj - newaj) < 0.0001) continue; 

				this->alphas(j) = newaj; //ZMIANA WEKTORA ALPHAS

				newai = ai + this->labels(i) * this->labels(j) * (aj - newaj);
				this->alphas(i) = newai; //ZMIANA WEKTORA ALPHAS

				b1 = this->bias - Ei - this->labels(i)*(newai - ai)*this->RBFKernel(this->data.row(i), this->data.row(i)) - this->labels(j)*(newaj - aj)*this->RBFKernel(this->data.row(i), this->data.row(j));
				b2 = this->bias - Ej - this->labels(i)*(newai - ai)*this->RBFKernel(this->data.row(i), this->data.row(j)) - this->labels(j)*(newaj - aj)*this->RBFKernel(this->data.row(j), this->data.row(j));

				this->bias = 0.5*(b1 + b2);
				if(newai > 0 && newai < C) this->bias = b1;
            	if(newaj > 0 && newaj < C) this->bias = b2;
				alphaFlag++;
			}//if(warunek na aktualizację wektora alphas)
		}//for(unsigned int i = 0; i < n; i++)
		iter++;
		// pokazuj postęp uczenia (można wyrzucić w przyszłości)
		std::cout<<"\nIteracja i = "<<iter<<". ";
		
		if(alphaFlag == 0) passes++;
		else {
			std::cout<<"Wektor alf zmieniony";
			passes = 0;	
		}
	}//while(passes < passLimit && iter < iterLimit)

	clock_t endTrain = clock();
	this->trainTime = (double) (endTrain-startTrain) / CLOCKS_PER_SEC * 1000.0;
	return;
}

SVM::SVM(const MatrixXd& data, const VectorXd labels, SVMOptions options){
	/*
	konstruktor z argumentami
	*/
	this->data = data;
	this->labels = labels;
	this->bias = 0.0;
	this->C = options.C;
	this->tol = options.tol;
	this->sigma = options.sigma;
	this->iterLimit = options.iterLimit;
	this->passLimit = options.passLimit;
	this->n = data.rows();
	this->d = data.cols();
	this->alphas = VectorXd::Zero(this->n); 
}

int SVM::randomWithExceptionOf(int randMin, int randMax, int exception){
	/*
	losuj liczbę z przedzialu randMin..randMax oprócz exception zawartego w tym przedziale
	*/
	int outcome = exception;
	while(outcome == exception){
		outcome = randMin + (rand() % (randMax + 1));
	}
	return outcome;
}

double SVM::RBFKernel(const VectorXd& v1, const VectorXd& v2){
	/*
	Radial Basis Function Kernel dokonujący przekształcenia
	przestrzeni wejściowej, co umożliwia poprawną klasyfikację próbek
	które nie są separowalne liniowo
	*/
	unsigned int v1len = v1.rows();
	unsigned int v2len = v2.rows();
	if(v1len == v2len){
		double sum = 0.0;
		for(unsigned int i = 0; i<v1len; i++){
			sum += (v1(i) - v2(i)) * (v1(i) - v2(i));
		}
		return exp(-sum/(2.0 * this->sigma * this->sigma));
	}
	else{
		std::cout<<"Incorrect data for RBF calculation, 0.0 returned"<<std::endl;
		return 0.0;
	}
}

double SVM::giveMargin(const VectorXd& v){
	/*
	oblicz położenie danego wektora danych od hiperpłaszczyzny rodzielającej
	w przestrzeni przekształconej funkcją RBF 
	jeśli bChanged > 0 -> próbka dodatnia (+) tu: N
	jeśli bChanged <= -> próbka ujemna (-) tu: V
	*/
	double bChanged = this->bias;
	for(unsigned int i = 0; i < this->n; i++){
		bChanged += this->alphas(i) * this->labels(i) * this->RBFKernel(v, this->data.row(i));
	}
	return bChanged;
}

int SVM::predict(const VectorXd& x){
	/*
	dokonaj predykcji przy użyciu metody giveMargin
	*/
	return (this->giveMargin(x) > 0) ? 1 : -1; 
}

VectorXd SVM::massPredict(const MatrixXd& data){
	/*
	wykonaj metodę predict w pętli dla macierzy danych ze zbioru testowego
	*/
	clock_t startTest = clock();
	unsigned int observations = data.rows();
	VectorXd result(observations);
	for(unsigned int i = 0; i < observations; i++){
		result(i) = this->predict(data.row(i));
	}
	clock_t endTest = clock();
	this->testTime = (double) (endTest-startTest) / CLOCKS_PER_SEC * 1000.0;
	return result;
}

void evaluateModel(const VectorXd& target, const VectorXd& outcome){
	/*
	oblicz ilość próbek poprawnie sklasyfikowanych
	*/
	unsigned int targetLen = target.rows();
	unsigned int outcomeLen = outcome.rows();
	unsigned int pos_pos = 0;
	unsigned int pos_neg = 0;
	unsigned int neg_pos = 0;
	unsigned int neg_neg = 0;
	if(targetLen == outcomeLen){
		for(unsigned int i = 0; i < targetLen; i++){
			if(target(i) == 1 && outcome(i) == 1) pos_pos++;
			else if(target(i) == 1 && outcome(i) == -1) pos_neg++;
			else if(target(i) == -1 && outcome(i) == 1) neg_pos++;
			else neg_neg++;
		}
		double successPercent = 100.00 * (pos_pos + neg_neg) / (pos_pos + neg_neg + pos_neg + neg_pos); 
		std::ofstream out("output.txt");
    	out << "Poprawnie sklasyfikowano " << successPercent << "% przykładów.\n";
    	out.close();
		return;
	}
	else{
		std::cout<<"Test set target and model outcome have different lengths."<<std::endl;
		return;
	}
}

void appendToOutputFile(double trainTime, double testTime){
	/*
	stwórz strumień do pliku output.txt zawierającego podstawowe
	parametry działania algorytmu
	*/
	std::ofstream out("output.txt", std::ofstream::app);
	std::cout<<"\n=================================\n";
	std::cout<<"Stworzono pomyslnie model\n";
	std::cout << "Trening zajal " << trainTime << " ms.\n";
    std::cout << "Testowanie zajelo " << testTime << " ms.";
	std::cout<<"\n=================================\n";
    out << "Uczenie modelu zajęło " << trainTime << " ms.\n";
    out << "Testowanie modelu zajęło " << testTime << " ms.";
    out.close();
    return;
}

void writeDecisionToCSV(const VectorXd& v){
	/*
	zapisz przyporządkowanie do pliku csv
	*/
	std::ofstream out("out.csv", std::ofstream::app);
	unsigned n = v.rows();
	for(unsigned int i = 0; i < n; i++){
    	out << v(i) <<"\n";
    }
    out.close();
    return;
}

ParsedData parseData(char** inputArray){
	/*
	na podstawie nazw pliku z konsoli zwróć strukturę (ParsedData) zawierającą
	2 macierze (dane do uczenia i testowe) i 2 wektory (poprawne przyporządkowanie
	danych do uczenia i testów)
	*/
	ParsedData dataToReturn;

	std::string trainFile = "./" + std::string(inputArray[1]);
	std::string testFile = "./" + std::string(inputArray[2]);
	
	MatrixXd trainSet = load_csv<MatrixXd>(trainFile);
	VectorXd trainSetOut = extractLabelsVec(trainSet);
	MatrixXd testSet = load_csv<MatrixXd>(testFile);
	VectorXd testSetOut = extractLabelsVec(testSet);

	dataToReturn.trainSet = trainSet;
	dataToReturn.testSet = testSet;
	dataToReturn.trainSetOut = trainSetOut;
	dataToReturn.testSetOut = testSetOut;

	return dataToReturn;
}

int main(int argc, char** argv){
	// wczytaj dane z plików, których nazwę podano w konsoli - założono, że pliki z danymi znajdują się w tym samym folderze co program
	ParsedData data = parseData(argv);

	// utwórz model, podano domyślne parametry tj. C(1.0), tol(0.0001), sigma(0.5), iterLimit(10000), passLimit(10)
	SVM model(data.trainSet, data.trainSetOut, SVMOptions());
	
	// trenuj model na danych podanych w konstruktorze
	model.train();
	
	// dokonaj klasyfikacji danych podanych w pliku testowym; zapisz wektor z klasyfikacją do pliku CSV o nazwie out.csv
	VectorXd decision = model.massPredict(data.testSet);
	writeDecisionToCSV(decision);
	
	// oceń działanie klasyfikatora (ilość poprawnie sklasyfikowanych ewolucji serca N/V) oraz czas uczenia i testów
	evaluateModel(data.testSetOut, decision);
	appendToOutputFile(model.trainTime, model.testTime);

	return 0;
}