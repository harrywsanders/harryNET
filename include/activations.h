#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <string>

class ActivationFunction {
public:
    virtual double compute(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::string getName() const = 0;
    virtual Eigen::VectorXd eigenCompute(const Eigen::VectorXd& x) const = 0;
    virtual Eigen::VectorXd eigenDerivative(const Eigen::VectorXd& x, int index) const = 0;
    virtual ~ActivationFunction() = default;
    virtual ActivationFunction* clone() const = 0;
    virtual bool operator==(const ActivationFunction& other) const {
        return getName() == other.getName();
    }
    virtual bool operator!=(const ActivationFunction& other) const {
        return !(*this == other);
    }
    //overload the = operator
    ActivationFunction& operator=(const ActivationFunction& other) {
        return *this;
    }
    
};


class ReLU : public ActivationFunction {
public:
    ActivationFunction* clone() const override {
        return new ReLU(*this);
    }
    double compute(double x) const override {
        return std::max(0.0, x);
    }

    double derivative(double x) const override {
        return x > 0.0 ? 1.0 : 0.0;
    }

    std::string getName() const override {
        return "ReLU";
    }

    Eigen::VectorXd eigenCompute(const Eigen::VectorXd& x) const override {
        return x.unaryExpr([](double v) { return ReLU().compute(v); });
    }

    Eigen::VectorXd eigenDerivative(const Eigen::VectorXd& x, int index) const override {
    index++;
    return x.unaryExpr([](double v) { return ReLU().derivative(v); });
    }
};

class Sigmoid : public ActivationFunction {
public:
    ActivationFunction* clone() const override {
        return new Sigmoid(*this);
    }
    double compute(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double derivative(double x) const override {
        double sigmoid_x = compute(x);
        return sigmoid_x * (1.0 - sigmoid_x);
    }

    std::string getName() const override {
        return "Sigmoid";
    }

    Eigen::VectorXd eigenCompute(const Eigen::VectorXd& x) const override {
        std::cout << "WARNING: You Probably don't mean to do this." << std::endl;
        return x.unaryExpr([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
    }

    Eigen::VectorXd eigenDerivative(const Eigen::VectorXd& x, int index) const override {
        std::cout << "WARNING: You Probably don't mean to do this." << std::endl;
        index++;
        return x.unaryExpr([](double v) {
            double sigmoid_x = 1.0 / (1.0 + std::exp(-v));
            return sigmoid_x * (1.0 - sigmoid_x);
        });
    }
};

class Tanh : public ActivationFunction {
public:
    ActivationFunction* clone() const override {
        return new Tanh(*this);
    }
    double compute(double x) const override {
        return std::tanh(x);
    }

    double derivative(double x) const override {
        return 1.0 - std::pow(std::tanh(x), 2);
    }

    std::string getName() const override {
        return "Tanh";
    }

    Eigen::VectorXd eigenCompute(const Eigen::VectorXd& x) const override {
        std::cout << "WARNING: You Probably don't mean to do this." << std::endl;
        return x.unaryExpr([](double v) { return std::tanh(v); });
    }

    Eigen::VectorXd eigenDerivative(const Eigen::VectorXd& x, int index) const override {
        std::cout << "WARNING: You Probably don't mean to do this." << std::endl;
        index++;
        return x.unaryExpr([](double v) { return 1.0 - std::pow(std::tanh(v), 2); });
    }

};

// Softmax (special case as it operates on vectors)
class Softmax : public ActivationFunction {
public:

    double compute(double x) const override {
        std::cout << "WARNING: You Probably don't mean to do this." << std::endl;
        x = 0.0;
        return x;
    }

    double derivative(double x) const override {
        std::cout << "WARNING: You Probably don't mean to do this." << std::endl;
        x = 0.0;
        return x;
    }

    ActivationFunction* clone() const override {
        return new Softmax(*this);
    }
    Eigen::VectorXd eigenCompute(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd exp_x = x.unaryExpr([](double v) { return std::exp(v); });
        double sum = exp_x.sum();
        return exp_x / sum;
    }

    Eigen::VectorXd eigenDerivative(const Eigen::VectorXd& softmax_output, int index) const override {
        Eigen::VectorXd derivative = Eigen::VectorXd::Zero(softmax_output.size());
        for (int i = 0; i < softmax_output.size(); ++i) {
            if (i == index) {
                derivative(i) = softmax_output(i) * (1.0 - softmax_output(i));
            } else {
                derivative(i) = -softmax_output(i) * softmax_output(index);
            }
        }
        return derivative;
    }

    std::string getName() const override {
        return "Softmax";
    }
};
