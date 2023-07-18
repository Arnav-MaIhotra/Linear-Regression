#include "cregress.cpp"
#include <iostream>

int main(void) {

  std::vector<std::vector<double>> X = {
        {1.1},
        {2.1},
        {3.1}
    };
  std::vector<double> Y = {11.0, 21.0, 31.0};

  LinearRegression model;
  model.fit(X, Y);

  std::vector<double> new_data = {6.1};
  double prediction = model.predict(new_data);

  std::cout << "Prediction: " << prediction << std::endl;

  return 0;
  
}