// Project 1 - Linear Regression
// This tool iteratively adjusts model weights to minimize loss using gradient descent.
// CSCI 581 - Spring 2024
// Daniel Hawkins
// Last modified: 02/12/24

#include <iomanip>
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::showpoint;
using std::vector;

class ModelData{
  public:
    ModelData();
    void initializeModel();
    void runIterations();
    void printResults();

  private:  
    int m, d, n;
    double alpha;
    vector<vector<int>> xAttributes;
    vector<int> yLabels;
    vector<double> weights;
    vector<double> hCalculations;

    void hCalculation();
    void recalculateWeights();
};

int main(){
  ModelData data;
  data.initializeModel();
  data.runIterations();
  data.printResults();
  return 0;
}

ModelData::ModelData() : m(0), d(0), n(0), alpha(0.0) {}

void ModelData::initializeModel(){
  cin >> m >> d >> n >> alpha;

  xAttributes.resize(m);
  yLabels.resize(m);
  weights.resize(d + 1, 1);

  for (int i = 0; i < m; i++){
    xAttributes[i].resize(d + 1);
  }

  for (int i = 0; i < m; i++){
    xAttributes[i][0] = 1;

    for (int j = 1; j < d + 1; j++){
      cin >> xAttributes[i][j];
    }

    cin >> yLabels[i];
  }
}

void ModelData::hCalculation(){
  hCalculations.resize(m);

  for (int i = 0; i < xAttributes.size(); i++){
    double hCalc = 0.0;

    for (int j = 0; j < xAttributes[i].size(); j++){
      hCalc += weights[j] * xAttributes[i][j];
    }

    hCalculations[i] = hCalc;
  }
}

void ModelData::recalculateWeights(){
  vector<double> tempWeights(weights.size(), 0.0);

  for (int i = 0; i < weights.size(); i++){
    double sumForWeight = 0.0;

    for (int j = 0; j < xAttributes.size(); j++){
      sumForWeight += xAttributes[j][i] * (yLabels[j] - hCalculations[j]);
    }

    tempWeights[i] = weights[i] + alpha * sumForWeight;
  }

  weights = tempWeights;
}

void ModelData::runIterations(){
  for (int i = 0; i < n; i++){
    hCalculation();
    recalculateWeights();
  }
}

void ModelData::printResults(){
  for (int i = 0; i < weights.size(); i++){
    cout << fixed << showpoint << setprecision(12) << weights[i] << endl;
  }
}