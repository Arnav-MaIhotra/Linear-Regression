#include <iostream>
#include <vector>
#include <random>
#include <cmath>

class LinearRegression {
private:
    double learning_rate;
    int max_iters;
    double tolerance;
    std::vector<double> params;
    double bias;

public:
    LinearRegression(double learning_rate = 0.001, int max_iters = 100000, double tolerance = 1e-10) {
        this->learning_rate = learning_rate;
        this->max_iters = max_iters;
        this->tolerance = tolerance;
        this->params = {};
        this->bias = 0.0;
    }

    double h(const std::vector<double>& X) {
        double result = 0.0;
        for (int i = 0; i < X.size(); i++) {
            result += X[i] * this->params[i];
        }
        result += this->bias;
        return result;
    }

    double j(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        double sum_squared_error = 0.0;
        for (int i = 0; i < X.size(); i++) {
            double predicted = this->h(X[i]);
            double error = predicted - y[i];
            sum_squared_error += pow(error, 2);
        }
        return sum_squared_error / (2 * X.size());
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, bool progress = false) {
        int num_features = X[0].size();
        int num_samples = X.size();

        this->params = std::vector<double>(num_features, 0.0);
        this->bias = 0.0;

        if (Y.size() != num_samples) {
            throw std::invalid_argument("Number of samples in X and Y do not match.");
        }

        int iters = 0;
        double prev_cost = INFINITY;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, num_samples - 1);

        for (int i = 0; i < this->max_iters; i++) {
            int index = dist(gen);
            std::vector<double> x = X[index];
            double y = Y[index];

            double predicted = this->h(x);

            double grad_coef = (2.0 / num_samples) * dotProduct(x, predicted - y);
            double grad_inte = (2.0 / num_samples) * (predicted - y);

            for (int j = 0; j < num_features; j++) {
                this->params[j] -= this->learning_rate * grad_coef * x[j];
            }
            this->bias -= this->learning_rate * grad_inte;

            double current_cost = this->j(X, Y);

            if (std::abs(current_cost - prev_cost) < this->tolerance) {
                break;
            }

            prev_cost = current_cost;

            if (progress) {
                std::cout << "Squared error: " << this->j(X, Y) << std::endl;
                iters++;
            }
        }

        if (progress) {
            std::cout << "Iterations: " << iters << std::endl;
        }
    }

    double predict(const std::vector<double>& x) {
        return this->h(x);
    }

private:
    double dotProduct(const std::vector<double>& a, const double b) {
        double result = 0.0;
        for (int i = 0; i < a.size(); i++) {
            result += a[i] * b;
        }
        return result;
    }
};