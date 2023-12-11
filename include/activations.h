#pragma once

#include <cmath>
#include <Eigen/Dense>

enum class ActivationFunctionType
{
    Sigmoid,
    ReLU,
    Tanh,
    Softmax
};

namespace ActivationFunctions
{
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoidDerivative(double x)
    {
        double sigmoid_x = sigmoid(x);
        return sigmoid_x * (1.0 - sigmoid_x);
    }

    double relu(double x)
    {
        return std::max(0.0, x);
    }

    double reluDerivative(double x)
    {
        return x > 0.0 ? 1.0 : 0.0;
    }

    double tanh(double x)
    {
        return std::tanh(x);
    }

    double tanhDerivative(double x)
    {
        return 1.0 - std::pow(std::tanh(x), 2);
    }

    Eigen::VectorXd softmax(const Eigen::VectorXd& x)
    {
        Eigen::VectorXd exp_x = x.unaryExpr([](double v) { return std::exp(v); });
        double sum = exp_x.sum();
        return exp_x / sum;
    }

    double softmaxDerivative(const Eigen::VectorXd& softmax_output, int index)
    {
        double s_i = softmax_output(index);
        double sum_derivative = 0.0;
        for (int j = 0; j < softmax_output.size(); ++j) {
            if (j == index) {
                sum_derivative += softmax_output(j) * (1.0 - softmax_output(j));
            } else {
                sum_derivative -= softmax_output(j) * softmax_output(index);
            }
        }
        return sum_derivative;
    }
}
