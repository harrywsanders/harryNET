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
    
    double calculateCrossEntropy(std::vector<Eigen::VectorXd> &inputs, std::vector<Eigen::VectorXd> &targetOutputs);

    Eigen::VectorXd predict(const Eigen::VectorXd &inputs);

    double accuracy(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targetOutputs);

    void save(const std::string &filename);

    void load(const std::string &filename);
};

void NeuralNetwork::forwardPropagate(const Eigen::VectorXd &inputs) {
    layers[0].output = inputs;

    for (size_t i = 1; i < layers.size(); i++) {
        Eigen::VectorXd z = layers[i].weights * layers[i - 1].output + layers[i].bias;

        layers[i].output = layers[i].activationFunc->apply(z);
    }
}

void NeuralNetwork::backPropagate(const Eigen::VectorXd &targetOutputs) {
    Layer &outputLayer = layers.back();
    outputLayer.delta.noalias() = outputLayer.output - targetOutputs;

    for (int i = layers.size() - 2; i >= 0; i--) {
        Layer &hiddenLayer = layers[i];
        Layer &nextLayer = layers[i + 1];

        hiddenLayer.delta.noalias() = nextLayer.weights.transpose() * nextLayer.delta;

        Eigen::VectorXd derivative = hiddenLayer.activationFunc->derivative(hiddenLayer.output, targetOutputs);
        hiddenLayer.delta = hiddenLayer.delta.cwiseProduct(derivative);
    }
}



void NeuralNetwork::updateWeightsAndBiases(double learningRate, int t, double lambda, double beta1, double beta2, double epsilon) {
    for (int i = 1; i < static_cast<int>(layers.size()); ++i) {
        Layer &layer = layers[i];
        const Eigen::MatrixXd &prev_output = layers[i - 1].output; 

        Eigen::MatrixXd weight_gradients = layer.delta * prev_output.transpose(); 
        weight_gradients.noalias() += lambda * layer.weights;

        layer.m_weights = beta1 * layer.m_weights + (1 - beta1) * weight_gradients;
        layer.v_weights = beta2 * layer.v_weights + (1 - beta2) * weight_gradients.array().square().matrix();

        Eigen::MatrixXd m_hat_weights = layer.m_weights / (1 - pow(beta1, t));
        Eigen::MatrixXd v_hat_weights = layer.v_weights / (1 - pow(beta2, t));

        layer.weights -= (learningRate * m_hat_weights.array() / (v_hat_weights.array().sqrt() + epsilon)).matrix();

        Eigen::VectorXd bias_gradients = layer.delta.rowwise().mean(); 

        layer.m_bias = beta1 * layer.m_bias + (1 - beta1) * bias_gradients;
        layer.v_bias = beta2 * layer.v_bias + (1 - beta2) * bias_gradients.array().square().matrix();

        Eigen::VectorXd m_hat_bias = layer.m_bias / (1 - pow(beta1, t));
        Eigen::VectorXd v_hat_bias = layer.v_bias / (1 - pow(beta2, t));

        layer.bias -= (learningRate * m_hat_bias.array() / (v_hat_bias.array().sqrt() + epsilon)).matrix();
    }
}


void NeuralNetwork::train(std::vector<Eigen::VectorXd> &trainInputs, std::vector<Eigen::VectorXd> &trainOutputs, std::vector<Eigen::VectorXd> &validInputs, std::vector<Eigen::VectorXd> &validOutputs, double learningRate, int nEpochs, int batchSize, int patience, double lambda, bool progressBar) {
    int t = 0;
    double bestValidMSE = std::numeric_limits<double>::max();
    int epochsNoImprove = 0;
    int totalSteps = (trainInputs.size() + batchSize - 1) / batchSize;

    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        int completedSteps = 0;
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(trainInputs.size()); i += batchSize) {
            processBatch(trainInputs, trainOutputs, i, batchSize);
            #pragma omp critical
            {
                completedSteps++;
                if (progressBar) {
                    printProgressBar(completedSteps, totalSteps);
                }
            }
        }

        for (int i = 1; i < static_cast<int>(layers.size()); ++i) {
            updateWeightsAndBiases(learningRate, t, lambda);
        }

        double validMSE = calculateCrossEntropy(validInputs, validOutputs);
        double trainMSE = calculateCrossEntropy(trainInputs, trainOutputs);
        if (progressBar) {
            std::cout << "\nEpoch: " << epoch << ", Train CE: " << trainMSE << ", Valid CE: " << validMSE << std::endl;
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

double NeuralNetwork::calculateCrossEntropy(std::vector<Eigen::VectorXd> &inputs, std::vector<Eigen::VectorXd> &targetOutputs) {
    if (inputs.empty()) return std::numeric_limits<double>::quiet_NaN();
    double crossEntropy = 0.0;
    for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
        forwardPropagate(inputs[i]); 
        Eigen::VectorXd output = layers.back().output; 
        crossEntropy += -targetOutputs[i].dot(output.unaryExpr([](double x) { return std::log(x); }));
    }
    return crossEntropy / inputs.size();
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
