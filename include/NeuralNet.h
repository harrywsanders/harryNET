/**
 * @class NeuralNetwork
 *
 * @brief A simple implementation of a feed-forward neural network using the backpropagation algorithm.
 *
 * This class represents a neural network model consisting of multiple layers of neurons. Each neuron has a set of weights, a bias, an output value, and a delta value used for backpropagation.
 * The neural network uses the sigmoid activation function for neurons and the mean squared error (MSE) loss function.
 *
 * The class provides methods for forward propagation (predicting outputs from inputs), backward propagation (computing gradients of the loss function), and updating the weights and biases of the neurons (learning from the gradients).
 * The train method allows to train the network on a dataset with a specified learning rate and number of epochs.
 * The predict method computes the outputs for a given set of inputs after the network has been trained.
 *
 * Early stopping is implemented in the train method, which halts training when validation loss hasn't improved for a number of epochs (specified by the 'patience' parameter).
 * The train method also implements data shuffling at the start of each epoch.
 *
 */

#include <chrono>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include "layers.h"
#include "activations.h"

using namespace ActivationFunctions;

/**
 * Helper function to show training progress bar in the console.
 */
void printProgressBar(int current, int total) {
    int percent = (current * 100) / total;
    int progressBarLength = percent / 2;
    std::string progressBar = std::string(progressBarLength, '=') + std::string(50 - progressBarLength, ' ');
    std::cout << "\rTraining progress: [" << progressBar << "] " << percent << "%   " << std::flush;
}


class NeuralNetwork
{
public:
    std::vector<Layer> layers;

    void forwardPropagate(const Eigen::VectorXd &inputs);

    void backPropagate(const Eigen::VectorXd &targetOutputs);

    void updateWeightsAndBiases(double learningRate, int t, double lambda, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void train(std::vector<Eigen::VectorXd> &trainInputs, std::vector<Eigen::VectorXd> &trainOutputs, std::vector<Eigen::VectorXd> &validInputs, std::vector<Eigen::VectorXd> &validOutputs, double learningRate, int nEpochs, int batchSize, int patience, double lambda, bool progressBar = true);

    void processBatch(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &outputs, int startIndex, int batchSize);

    double calculateMSE(std::vector<Eigen::VectorXd> &inputs, std::vector<Eigen::VectorXd> &targetOutputs);

    Eigen::VectorXd predict(const Eigen::VectorXd &inputs);

    double accuracy(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targetOutputs);

    void save(const std::string &filename);

    void load(const std::string &filename);
};

void NeuralNetwork::forwardPropagate(const Eigen::VectorXd &inputs) {
    layers[0].output = inputs;
    for (size_t i = 1; i < layers.size(); i++) {
        layers[i].output.noalias() = layers[i].weights * layers[i - 1].output + layers[i].bias;

        if (layers[i].activationFunction == ActivationFunctionType::Sigmoid) {
            layers[i].output = layers[i].output.unaryExpr(&sigmoid);
        } else if (layers[i].activationFunction == ActivationFunctionType::ReLU) {
            layers[i].output = layers[i].output.unaryExpr(&relu);
        } else if (layers[i].activationFunction == ActivationFunctionType::Softmax) {
            layers[i].output = layers[i].output.unaryExpr(&softmax);
        }
    }
}

void NeuralNetwork::backPropagate(const Eigen::VectorXd &targetOutputs) {
    Layer &outputLayer = layers.back();
    outputLayer.delta.noalias() = (outputLayer.output - targetOutputs).cwiseProduct(outputLayer.output).cwiseProduct(Eigen::VectorXd::Ones(outputLayer.output.size()) - outputLayer.output);

    for (int i = layers.size() - 2; i >= 0; i--) {
        Layer &hiddenLayer = layers[i];
        Layer &nextLayer = layers[i + 1];
        hiddenLayer.delta.noalias() = nextLayer.weights.transpose() * nextLayer.delta;
        hiddenLayer.delta = hiddenLayer.delta.cwiseProduct(hiddenLayer.output).cwiseProduct(Eigen::VectorXd::Ones(hiddenLayer.output.size()) - hiddenLayer.output);
    }
}

void NeuralNetwork::updateWeightsAndBiases(double learningRate, int t, double lambda, double beta1, double beta2, double epsilon) {
    double beta1_pow_t = std::pow(beta1, t);
    double beta2_pow_t = std::pow(beta2, t);

    for (int i = 1; i < static_cast<int>(layers.size()); ++i) {
        Layer &layer = layers[i];
        Eigen::MatrixXd weight_gradients = layer.delta * layers[i - 1].output.transpose();

        // Add L2 regularization term
        weight_gradients.noalias() += lambda * layer.weights;

        layer.m_weights = beta1 * layer.m_weights + (1 - beta1) * weight_gradients;
        layer.v_weights = beta2 * layer.v_weights.array() + (1 - beta2) * weight_gradients.array().square();

        // Bias correction for weights
        Eigen::MatrixXd m_weights_hat = layer.m_weights.array() / (1 - std::pow(beta1, t));
        Eigen::MatrixXd v_weights_hat = layer.v_weights.array() / (1 - std::pow(beta2, t));


        // Update weights
        layer.weights = layer.weights.array() - learningRate * m_weights_hat.array() / (v_weights_hat.array().sqrt() + epsilon);

        // Calculate the first and second moment for biases
        layer.m_bias = beta1 * layer.m_bias.array() + (1 - beta1) * layer.delta.array();
        layer.v_bias = beta2 * layer.v_bias.array() + (1 - beta2) * layer.delta.array().square();

        // Bias correction for biases
        Eigen::VectorXd m_bias_hat = layer.m_bias.array() / (1 - beta1_pow_t);
        Eigen::VectorXd v_bias_hat = layer.v_bias.array() / (1 - beta2_pow_t);

        // Update biases
        layer.bias = layer.bias.array() - learningRate * m_bias_hat.array() / (v_bias_hat.array().sqrt() + epsilon);
    }
}

void NeuralNetwork::train(std::vector<Eigen::VectorXd> &trainInputs, std::vector<Eigen::VectorXd> &trainOutputs, std::vector<Eigen::VectorXd> &validInputs, std::vector<Eigen::VectorXd> &validOutputs, double learningRate, int nEpochs, int batchSize, int patience, double lambda, bool progressBar) {
    int t = 0;
    double bestValidMSE = std::numeric_limits<double>::max();
    int epochsNoImprove = 0;
    int totalSteps = (trainInputs.size() + batchSize - 1) / batchSize; // Total steps in one epoch

    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        for (int i = 0; i < static_cast<int>(trainInputs.size()); i += batchSize) {
            processBatch(trainInputs, trainOutputs, i, batchSize);
            t++;
            updateWeightsAndBiases(learningRate, t, lambda);
            if (progressBar) {
                printProgressBar(t, totalSteps);
            }
        }

        double validMSE = calculateMSE(validInputs, validOutputs);
        double trainMSE = calculateMSE(trainInputs, trainOutputs);
        if (progressBar) {
            std::cout << "\nEpoch: " << epoch << ", Train MSE: " << trainMSE << ", Valid MSE: " << validMSE << std::endl;
        }

        if (validMSE < bestValidMSE) {
            bestValidMSE = validMSE;
            epochsNoImprove = 0;
        } else {
            epochsNoImprove++;
        }

        if (epochsNoImprove >= patience) {
            std::cout << "Early stopping at epoch: " << epoch << std::endl;
            break;
        }
        t = 0;
    }
}

void NeuralNetwork::processBatch(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &outputs, int startIndex, int batchSize) {
    int endIndex = std::min(startIndex + batchSize, static_cast<int>(inputs.size()));
    for (int j = startIndex; j < endIndex; ++j) {
        forwardPropagate(inputs[j]);
        backPropagate(outputs[j]);
    }
}


double NeuralNetwork::calculateMSE(std::vector<Eigen::VectorXd> &inputs, std::vector<Eigen::VectorXd> &targetOutputs) {
    if (inputs.empty()) return std::numeric_limits<double>::quiet_NaN();
    double mse = 0.0;
    for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
        forwardPropagate(inputs[i]); 
        Eigen::VectorXd output = layers.back().output; 
        mse += (output - targetOutputs[i]).array().square().mean();
    }
    return mse / inputs.size();
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &inputs)
{
    forwardPropagate(inputs);
    return layers.back().output;
}



double NeuralNetwork::accuracy(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targetOutputs) {
    int correctCount = 0;
    for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
        forwardPropagate(inputs[i]); // Directly use forward propagation
        Eigen::VectorXd output = layers.back().output; // Reuse the output from forward propagation
        int predictedClass = std::max_element(output.data(), output.data() + output.size()) - output.data();
        int actualClass = std::max_element(targetOutputs[i].data(), targetOutputs[i].data() + targetOutputs[i].size()) - targetOutputs[i].data();
        if (predictedClass == actualClass) {
            correctCount++;
        }
    }
    return static_cast<double>(correctCount) / inputs.size();
}


void NeuralNetwork::save(const std::string &filename)
{
    std::ofstream outputFile(filename, std::ios::binary);

    if (!outputFile.is_open())
    {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    int numLayers = layers.size();
    outputFile.write(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));

    for (const auto &layer : layers)
    {
        int rows = layer.weights.rows();
        int cols = layer.weights.cols();

        outputFile.write(reinterpret_cast<char *>(&rows), sizeof(rows));
        outputFile.write(reinterpret_cast<char *>(&cols), sizeof(cols));

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                double weight = layer.weights(i, j);
                outputFile.write(reinterpret_cast<char *>(&weight), sizeof(weight));
            }
        }

        for (int i = 0; i < rows; ++i)
        {
            double bias = layer.bias(i);
            outputFile.write(reinterpret_cast<char *>(&bias), sizeof(bias));
        }
    }

    outputFile.close();
}

void NeuralNetwork::load(const std::string &filename)
{
    std::ifstream inputFile(filename, std::ios::binary);

    if (!inputFile.is_open())
    {
        throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    int numLayers;
    inputFile.read(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));

    layers.resize(numLayers);

    for (auto &layer : layers)
    {
        int rows, cols;
        inputFile.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        inputFile.read(reinterpret_cast<char *>(&cols), sizeof(cols));

        layer.weights.resize(rows, cols);
        layer.bias.resize(rows);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                double weight;
                inputFile.read(reinterpret_cast<char *>(&weight), sizeof(weight));
                layer.weights(i, j) = weight;
            }
        }

        for (int i = 0; i < rows; ++i)
        {
            double bias;
            inputFile.read(reinterpret_cast<char *>(&bias), sizeof(bias));
            layer.bias(i) = bias;
        }
    }

    inputFile.close();
}
