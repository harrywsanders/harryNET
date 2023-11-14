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
#pragma once

enum class LayerType
{
    Dense,
    Convolutional
};

class Layer {
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

    Layer() : type(LayerType::Dense) {}

    Layer(size_t nNeurons, size_t nInputsPerNeuron, LayerType layerType = LayerType::Dense)
        : type(layerType)
    {
        if (nNeurons == 0 || nInputsPerNeuron == 0) {
            throw std::invalid_argument("Number of neurons and inputs per neuron must be greater than 0.");
        }

        if (layerType == LayerType::Dense) {
            initializeDenseLayer(nNeurons, nInputsPerNeuron);
        } else {
            throw std::invalid_argument("Unsupported layer type for this constructor.");
        }
    }

    Layer(size_t nFilters, size_t fSize, size_t s, size_t p, LayerType layerType = LayerType::Convolutional)
        : nFilters(nFilters), filterSize(fSize), stride(s), padding(p), type(layerType)
    {
        if (nFilters == 0 || filterSize == 0) {
            throw std::invalid_argument("Number of filters and filter size must be greater than 0.");
        }

        if (layerType == LayerType::Convolutional) {
            initializeConvolutionalLayer();
        } else {
            throw std::invalid_argument("Unsupported layer type for this constructor.");
        }
    }

private:
    void initializeDenseLayer(size_t nNeurons, size_t nInputsPerNeuron) {
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

        for (size_t i = 0; i < nNeurons; i++) {
            for (size_t j = 0; j < nInputsPerNeuron; j++) {
                weights(i, j) = distribution(generator);
            }
            bias[i] = distribution(generator);
        }
    }

    void initializeConvolutionalLayer() {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
        distribution = std::normal_distribution<double>(0.0, std::sqrt(2.0 / (filterSize * filterSize)));

        for (size_t i = 0; i < nFilters; i++) {
            filters.push_back(Eigen::MatrixXd::Random(filterSize, filterSize));
            m_filters.push_back(Eigen::MatrixXd::Zero(filterSize, filterSize));
            v_filters.push_back(Eigen::MatrixXd::Zero(filterSize, filterSize));
            filterBias.push_back(distribution(generator));
            m_filterBias.push_back(0.0);
            v_filterBias.push_back(0.0);
        }
    }
};
