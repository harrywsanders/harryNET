#pragma once
#include <string>
#include "Eigen/Dense"

class ActivationFunction
{
public:
    virtual std::unique_ptr<ActivationFunction> clone() const = 0;
    virtual ~ActivationFunction() = default;
    virtual Eigen::VectorXd apply(const Eigen::VectorXd &input) const = 0;
    virtual Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &target) const = 0;
    virtual std::string getName() const = 0;

};

class inputActivation : public ActivationFunction
{
public:
    std::unique_ptr<ActivationFunction> clone() const override
    {
        return std::make_unique<inputActivation>(*this);
    }
    Eigen::VectorXd apply(const Eigen::VectorXd &input) const override
    {
        return input;
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &target) const override
    {
        (void)target;
        return Eigen::VectorXd::Ones(input.size());
    }

    std::string getName() const override
    {
        return "InputLayer";
    }
};

class Sigmoid : public ActivationFunction
{
public:
    std::unique_ptr<ActivationFunction> clone() const override
    {
        return std::make_unique<Sigmoid>(*this);
    }
    Eigen::VectorXd apply(const Eigen::VectorXd &input) const override
    {
        return 1.0 / (1.0 + (-input.array()).exp());
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &target) const override
    {
        (void)target;
        Eigen::VectorXd sigmoid = apply(input);
        return sigmoid.array() * (1 - sigmoid.array());
    }

    std::string getName() const override
    {
        return "Sigmoid";
    }
};

class Tanh : public ActivationFunction
{
public:
    std::unique_ptr<ActivationFunction> clone() const override
    {
        return std::make_unique<Tanh>(*this);
    }
    Eigen::VectorXd apply(const Eigen::VectorXd &input) const override
    {
        return input.array().tanh();
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &target) const override
    {
        (void)target;
        return 1.0 - input.array().tanh().square();
    }

    std::string getName() const override
    {
        return "Tanh";
    }
};

class ReLU : public ActivationFunction
{
public:
    std::unique_ptr<ActivationFunction> clone() const override
    {
        return std::make_unique<ReLU>(*this);
    }
    Eigen::VectorXd apply(const Eigen::VectorXd &input) const override
    {
        return input.cwiseMax(0.0);
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &target) const override
    {
        (void)target;
        return input.unaryExpr([](double x)
                               { return x > 0 ? 1.0 : 0.0; });
    }

    std::string getName() const override
    {
        return "ReLU";
    }
};

class LeakyReLU : public ActivationFunction
{
public:
    std::unique_ptr<ActivationFunction> clone() const override
    {
        return std::make_unique<LeakyReLU>(*this);
    }
    Eigen::VectorXd apply(const Eigen::VectorXd &input) const override
    {
        return input.array().max(0.01 * input.array());
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &target) const override
    {
        (void)target;
        return input.unaryExpr([](double x)
                               { return x > 0 ? 1.0 : 0.01; });
    }

    std::string getName() const override
    {
        return "LeakyReLU";
    }
};

class Softmax : public ActivationFunction
{
public:
    std::unique_ptr<ActivationFunction> clone() const override
    {
        return std::make_unique<Softmax>(*this);
    }
    Eigen::VectorXd apply(const Eigen::VectorXd &input) const override
    {
        Eigen::VectorXd exp = input.array().exp();
        return exp / exp.sum();
    }
    Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &target) const override
    {
        return apply(input) - target;
    }
    std::string getName() const override
    {
        return "Softmax";
    }
};