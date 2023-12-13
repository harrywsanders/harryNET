/**
 * @class Layer
 *
 * @brief Represents a layer of neurons in a neural network.
 *
 * This class is essentially a container for a set of neurons, but unlike the previous version, it does not use a `Neuron` class. Instead, it integrates the functionality of the `Neuron` directly into the `Layer` class.
 *
 * It provides a way to group neurons together into a layer, which simplifies the implementation of the neural network. Each neuron's weights, bias, output, delta, and moments for the Adam optimizer are stored in Eigen matrices and vectors, which allows for efficient linear algebra operations.
 *
 * The weights and biases for each neuron in the layer are initialized using He initialization.
 *
 * Note: This class does not provide methods for computations on the layer (such as computing outputs of neurons in the layer or updating their weights). These computations are done in the NeuralNetwork class.
 */
#include "Eigen/Dense"
#include <chrono>
#include <random>
#include <stdexcept>
#include "activations.h"
#include "activations.h"
#pragma once

enum class LayerType
{
    Dense,
    Convolutional
};



class Layer
{
public:
    Eigen::MatrixXd weights, m_weights, v_weights;
    Eigen::VectorXd bias, m_bias, v_bias, output, delta;
    std::vector<Eigen::MatrixXd> filters, m_filters, v_filters;
    std::vector<double> filterBias, m_filterBias, v_filterBias;
    size_t nFilters, filterSize;
    size_t stride, padding;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
    LayerType type;
    std::unique_ptr<ActivationFunction> activationFunction;

    //add a move constructor
    Layer(Layer&& other) noexcept
        : weights(std::move(other.weights)), m_weights(std::move(other.m_weights)), v_weights(std::move(other.v_weights)),
          bias(std::move(other.bias)), m_bias(std::move(other.m_bias)), v_bias(std::move(other.v_bias)),
          output(std::move(other.output)), delta(std::move(other.delta)),
          filters(std::move(other.filters)), m_filters(std::move(other.m_filters)), v_filters(std::move(other.v_filters)),
          filterBias(std::move(other.filterBias)), m_filterBias(std::move(other.m_filterBias)), v_filterBias(std::move(other.v_filterBias)),
          nFilters(std::move(other.nFilters)), filterSize(std::move(other.filterSize)),
          stride(std::move(other.stride)), padding(std::move(other.padding)),
          generator(std::move(other.generator)), distribution(std::move(other.distribution)),
          type(std::move(other.type)), activationFunction(std::move(other.activationFunction))
    {
    }

    //add a move assignment operator
    Layer& operator=(Layer&& other) noexcept
    {
        weights = std::move(other.weights);
        m_weights = std::move(other.m_weights);
        v_weights = std::move(other.v_weights);
        bias = std::move(other.bias);
        m_bias = std::move(other.m_bias);
        v_bias = std::move(other.v_bias);
        output = std::move(other.output);
        delta = std::move(other.delta);
        filters = std::move(other.filters);
        m_filters = std::move(other.m_filters);
        v_filters = std::move(other.v_filters);
        filterBias = std::move(other.filterBias);
        m_filterBias = std::move(other.m_filterBias);
        v_filterBias = std::move(other.v_filterBias);
        nFilters = std::move(other.nFilters);
        filterSize = std::move(other.filterSize);
        stride = std::move(other.stride);
        padding = std::move(other.padding);
        generator = std::move(other.generator);
        distribution = std::move(other.distribution);
        type = std::move(other.type);
        activationFunction = std::move(other.activationFunction);
        return *this;
    }

    //Copy constructor
    Layer(const Layer& other)
        : weights(other.weights), m_weights(other.m_weights), v_weights(other.v_weights),
          bias(other.bias), m_bias(other.m_bias), v_bias(other.v_bias),
          output(other.output), delta(other.delta),
          filters(other.filters), m_filters(other.m_filters), v_filters(other.v_filters),
          filterBias(other.filterBias), m_filterBias(other.m_filterBias), v_filterBias(other.v_filterBias),
          nFilters(other.nFilters), filterSize(other.filterSize),
          stride(other.stride), padding(other.padding),
          generator(other.generator), distribution(other.distribution),
          type(other.type), activationFunction(other.activationFunction->clone())
    {
    }

    //Copy assignment operator
    Layer& operator=(const Layer& other)
    {
        weights = other.weights;
        m_weights = other.m_weights;
        v_weights = other.v_weights;
        bias = other.bias;
        m_bias = other.m_bias;
        v_bias = other.v_bias;
        output = other.output;
        delta = other.delta;
        filters = other.filters;
        m_filters = other.m_filters;
        v_filters = other.v_filters;
        filterBias = other.filterBias;
        m_filterBias = other.m_filterBias;
        v_filterBias = other.v_filterBias;
        nFilters = other.nFilters;
        filterSize = other.filterSize;
        stride = other.stride;
        padding = other.padding;
        generator = other.generator;
        distribution = other.distribution;
        type = other.type;
        activationFunction = std::unique_ptr<ActivationFunction>(other.activationFunction->clone());
        return *this;
    }

    Layer() : type(LayerType::Dense), activationFunction(std::make_unique<ReLU>()) {}

    Layer(size_t nNeurons, size_t nInputsPerNeuron, LayerType layerType)
        : type(layerType), activationFunction(std::make_unique<ReLU>())
    {
        if (nNeurons == 0 || nInputsPerNeuron == 0)
        {
            throw std::invalid_argument("Number of neurons and inputs per neuron must be greater than 0.");
        }

        if (layerType == LayerType::Dense)
        {
            initializeDenseLayer(nNeurons, nInputsPerNeuron);
        }
        else
        {
            throw std::invalid_argument("Unsupported layer type for this constructor.");
        }
    }
    Layer(size_t nNeurons, size_t nInputsPerNeuron, LayerType layerType, std::unique_ptr<ActivationFunction> actFunction)
        : type(layerType), activationFunction(actFunction ? std::move(actFunction) : std::make_unique<ReLU>())
    {
        if (nNeurons == 0 || nInputsPerNeuron == 0)
        {
            throw std::invalid_argument("Number of neurons and inputs per neuron must be greater than 0.");
        }

        if (layerType == LayerType::Dense)
        {
            initializeDenseLayer(nNeurons, nInputsPerNeuron);
        }
        else
        {
            throw std::invalid_argument("Unsupported layer type for this constructor.");
        }
    }

    Layer(size_t nFilters, size_t fSize, size_t s, size_t p, LayerType layerType, std::unique_ptr<ActivationFunction> actFunction = nullptr)
        : nFilters(nFilters), filterSize(fSize), stride(s), padding(p), type(layerType), activationFunction(actFunction ? std::move(actFunction) : std::make_unique<ReLU>())
    {
        if (nFilters == 0 || filterSize == 0)
        {
            throw std::invalid_argument("Number of filters and filter size must be greater than 0.");
        }

        if (layerType == LayerType::Convolutional)
        {
            initializeConvolutionalLayer();
        }
        else
        {
            throw std::invalid_argument("Unsupported layer type for this constructor.");
        }
    }

private:
    void initializeDenseLayer(size_t nNeurons, size_t nInputsPerNeuron)
    {
        weights.resize(nNeurons, nInputsPerNeuron);
        m_weights = Eigen::MatrixXd::Zero(nNeurons, nInputsPerNeuron);
        v_weights = Eigen::MatrixXd::Zero(nNeurons, nInputsPerNeuron);
        bias.resize(nNeurons);
        m_bias = Eigen::VectorXd::Zero(nNeurons);
        v_bias = Eigen::VectorXd::Zero(nNeurons);
        output.resize(nNeurons);
        delta.resize(nNeurons);

        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
        distribution = std::normal_distribution<double>(0.0, std::sqrt(2.0 / nInputsPerNeuron));

        for (size_t i = 0; i < nNeurons; i++)
        {
            for (size_t j = 0; j < nInputsPerNeuron; j++)
            {
                weights(i, j) = distribution(generator);
            }
            bias[i] = distribution(generator);
        }
    }

    void initializeConvolutionalLayer()
    {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
        distribution = std::normal_distribution<double>(0.0, std::sqrt(2.0 / (filterSize * filterSize)));

        for (size_t i = 0; i < nFilters; i++)
        {
            filters.push_back(Eigen::MatrixXd::Random(filterSize, filterSize));
            m_filters.push_back(Eigen::MatrixXd::Zero(filterSize, filterSize));
            v_filters.push_back(Eigen::MatrixXd::Zero(filterSize, filterSize));
            filterBias.push_back(distribution(generator));
            m_filterBias.push_back(0.0);
            v_filterBias.push_back(0.0);
        }
    }
};
