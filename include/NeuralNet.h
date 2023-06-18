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

/**
 * Helper function to show training progress bar in the console.
 */
void printProgressBar(int current, int total)
{
    int percent = (current * 100) / total;
    std::cout << "\rTraining progress: [";

    int i = 0;
    for (; i < percent / 2; i++)
        std::cout << "=";

    for (; i < 50; i++)
        std::cout << " ";

    std::cout << "] " << percent << "%   ";

    std::cout.flush();
}

class NeuralNetwork
{
public:
    std::vector<Layer> layers;

    void forwardPropagate(const Eigen::VectorXd &inputs);

    void backPropagate(Eigen::VectorXd &targetOutputs);

    void updateWeightsAndBiases(double learningRate, int t, double lambda, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);  
    
    void train(std::vector<Eigen::VectorXd> &trainInputs, std::vector<Eigen::VectorXd> &trainOutputs, std::vector<Eigen::VectorXd> &validInputs, std::vector<Eigen::VectorXd> &validOutputs, double learningRate, int nEpochs, int batchSize, int patience, double lambda);

    double calculateMSE(std::vector<Eigen::VectorXd> &inputs, std::vector<Eigen::VectorXd> &targetOutputs);

    Eigen::VectorXd predict(const Eigen::VectorXd &inputs);

    double accuracy(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targetOutputs);

    void save(const std::string &filename);

    void load(const std::string &filename);
};

void NeuralNetwork::forwardPropagate(const Eigen::VectorXd &inputs)
{
    layers[0].output = inputs;
    for (size_t i = 1; i < layers.size(); i++)
    {
        layers[i].output = layers[i].weights * layers[i - 1].output + layers[i].bias;

        // apply sigmoid activation function
        layers[i].output = 1.0 / (1.0 + (-layers[i].output).array().exp());
    }
}

void NeuralNetwork::backPropagate(Eigen::VectorXd &targetOutputs)
{
    // calculate delta for output layer

    Layer &outputLayer = layers.back();
    outputLayer.delta = outputLayer.output.array() - targetOutputs.array();
    // calculate delta for hidden layers
    for (int i = layers.size() - 2; i >= 0; i--)
    {
        Layer &hiddenLayer = layers[i];
        Layer &nextLayer = layers[i + 1];
        hiddenLayer.delta = (nextLayer.weights.transpose() * nextLayer.delta).array() * (hiddenLayer.output.array() * (1.0 - hiddenLayer.output.array()));
    }
}

void NeuralNetwork::updateWeightsAndBiases(double learningRate, int t, double beta1, double beta2, double epsilon, double lambda)
{
    for (int i = 1; i < static_cast<int>(layers.size()); ++i)
    {
        Layer &layer = layers[i];
        Eigen::MatrixXd weight_gradients = layer.delta * layers[i - 1].output.transpose(); // Use output from previous layer

        // Add L2 regularization term
        weight_gradients += lambda * layer.weights;

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
        Eigen::VectorXd m_bias_hat = layer.m_bias.array() / (1 - std::pow(beta1, t));
        Eigen::VectorXd v_bias_hat = layer.v_bias.array() / (1 - std::pow(beta2, t));

        // Update biases
        layer.bias = layer.bias.array() - learningRate * m_bias_hat.array() / (v_bias_hat.array().sqrt() + epsilon);
    }
}

void NeuralNetwork::train(std::vector<Eigen::VectorXd> &trainInputs, std::vector<Eigen::VectorXd> &trainOutputs, std::vector<Eigen::VectorXd> &validInputs, std::vector<Eigen::VectorXd> &validOutputs, double learningRate, int nEpochs, int batchSize, int patience, double lambda)
{

    int t = 0;
    double bestValidMSE = std::numeric_limits<double>::max();
    int epochsNoImprove = 0;
    int totalSteps = (trainInputs.size() + batchSize - 1) / batchSize; // Total steps in one epoch

    for (int epoch = 0; epoch < nEpochs; epoch++)
    {
        for (int i = 0; i < static_cast<int>(trainInputs.size()); i += batchSize)
        {
            for (int j = i; j < i + batchSize && j < static_cast<int>(trainInputs.size()); j++)
            {
                forwardPropagate(trainInputs[j]);
                backPropagate(trainOutputs[j]);
            }
            t += 1;
            updateWeightsAndBiases(learningRate, t, lambda);
            printProgressBar(t, totalSteps);
        }

        double validMSE = calculateMSE(validInputs, validOutputs);
        double trainMSE = calculateMSE(trainInputs, trainOutputs);
        std::cout << "\nEpoch: " << epoch << ", Train MSE: " << trainMSE << ", Valid MSE: " << validMSE << std::endl;

        if (validMSE < bestValidMSE)
        {
            bestValidMSE = validMSE;
            epochsNoImprove = 0;
        }
        else
        {
            epochsNoImprove += 1;
        }

        if (epochsNoImprove == patience)
        {
            std::cout << "Early stopping at epoch: " << epoch << std::endl;
            break;
        }
        t = 0; // Reset the counter for the next epoch
    }
}

double NeuralNetwork::calculateMSE(std::vector<Eigen::VectorXd> &inputs, std::vector<Eigen::VectorXd> &targetOutputs)
{

    double mse = 0.0;
    for (int i = 0; i < static_cast<int>(inputs.size()); i++)
    {
        Eigen::VectorXd output = predict(inputs[i]);
        mse += (output - targetOutputs[i]).array().square().mean();
    }
    return mse / inputs.size();
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &input)
{

    forwardPropagate(input);
    return layers.back().output;
}

double NeuralNetwork::accuracy(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targetOutputs)
{

    int correctCount = 0;
    for (int i = 0; i < static_cast<int>(inputs.size()); i++)
    {
        Eigen::VectorXd output = predict(inputs[i]);
        int predictedClass = std::distance(output.data(), std::max_element(output.data(), output.data() + output.size()));
        int actualClass = std::distance(targetOutputs[i].data(), std::max_element(targetOutputs[i].data(), targetOutputs[i].data() + targetOutputs[i].size()));
        if (predictedClass == actualClass)
        {
            correctCount++;
        }
    }
    return static_cast<double>(correctCount) / inputs.size();
}
