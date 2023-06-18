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

class Layer {
public:
    Eigen::MatrixXd weights, m_weights, v_weights;
    Eigen::VectorXd bias, m_bias, v_bias, output, delta;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    Layer(size_t nNeurons, size_t nInputsPerNeuron)
        : weights(nNeurons, nInputsPerNeuron),
          m_weights(Eigen::MatrixXd::Zero(nNeurons, nInputsPerNeuron)),
          v_weights(Eigen::MatrixXd::Zero(nNeurons, nInputsPerNeuron)),
          bias(nNeurons),
          m_bias(Eigen::VectorXd::Zero(nNeurons)),
          v_bias(Eigen::VectorXd::Zero(nNeurons)),
          output(nNeurons),
          delta(nNeurons),
          generator(std::chrono::system_clock::now().time_since_epoch().count()),
          distribution(0.0, std::sqrt(2.0 / nInputsPerNeuron))
    {
        for (size_t i = 0; i < nNeurons; i++) {
            for (size_t j = 0; j < nInputsPerNeuron; j++) {
                weights(i, j) = distribution(generator);
            }
            bias[i] = distribution(generator);
        }
    }
};
