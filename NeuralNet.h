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

class NeuralNetwork
{
public:
    std::vector<Layer> layers;

    void forwardPropagate(const std::vector<double> &inputs);

    void backPropagate(std::vector<double> &targetOutputs);

    void updateWeightsAndBiases(double learningRate);

    void train(std::vector<std::vector<double>> &trainInputs, std::vector<std::vector<double>> &trainOutputs, std::vector<std::vector<double>> &validInputs, std::vector<std::vector<double>> &validOutputs, double learningRate, int nEpochs, int patience);

    double calculateMSE(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targetOutputs);

    std::vector<double> predict(const std::vector<double> &inputs);

    double accuracy(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targetOutputs);
};

// Implementations begin here:

void NeuralNetwork::forwardPropagate(const std::vector<double> &inputsConst)
{
    std::vector<double> inputs = inputsConst;
    for (auto &layer : layers)
    {
        std::vector<double> outputs;
        for (auto &neuron : layer.neurons)
        {
            double activation = neuron.bias;
            for (int i = 0; i < static_cast<int>(neuron.weights.size()); i++)
            {
                activation += neuron.weights[i] * inputs[i];
            }
            neuron.output = 1 / (1 + std::exp(-activation)); // sigmoid activation function
            outputs.push_back(neuron.output);
        }
        inputs = outputs; // outputs of this layer are inputs to the next layer
    }
}

void NeuralNetwork::backPropagate(std::vector<double> &targetOutputs)
{
    // Calculate output layer deltas
    Layer &outputLayer = layers.back();
    for (int i = 0; i < static_cast<int>(outputLayer.neurons.size()); i++)
    {
        double output = outputLayer.neurons[i].output;
        double target = targetOutputs[i];
        double error = target - output;
        outputLayer.neurons[i].delta = error * output * (1 - output); // derivative of MSE loss with respect to output * derivative of sigmoid
    }

    // Calculate hidden layer deltas
    for (int l = layers.size() - 2; l >= 0; l--)
    {
        Layer &hiddenLayer = layers[l];
        Layer &nextLayer = layers[l + 1];
        for (int i = 0; i < static_cast<int>(hiddenLayer.neurons.size()); i++)
        {
            double output = hiddenLayer.neurons[i].output;
            double error = 0.0;
            for (int j = 0; j < static_cast<int>(nextLayer.neurons.size()); j++)
            {
                error += nextLayer.neurons[j].delta * nextLayer.neurons[j].weights[i]; // weights from hidden layer to output layer
            }
            hiddenLayer.neurons[i].delta = error * output * (1 - output); // derivative of sigmoid
        }
    }
}

void NeuralNetwork::updateWeightsAndBiases(double learningRate)
{
    std::vector<double> inputs;
    if (layers.size() > 1)
    {
        for (auto &neuron : layers[layers.size() - 2].neurons)
        {
            inputs.push_back(neuron.output);
        }
    }

    for (auto &layer : layers)
    {
        for (auto &neuron : layer.neurons)
        {
            for (int i = 0; i < static_cast<int>(neuron.weights.size()); i++)
            {
                neuron.weights[i] += learningRate * neuron.delta * inputs[i]; // update weight
            }
            neuron.bias += learningRate * neuron.delta; // update bias
        }
    }
}

void NeuralNetwork::train(std::vector<std::vector<double>> &trainInputs, std::vector<std::vector<double>> &trainOutputs, std::vector<std::vector<double>> &validInputs, std::vector<std::vector<double>> &validOutputs, double learningRate, int nEpochs, int patience)
{
    double bestValidLoss = std::numeric_limits<double>::max();
    int epochsWithoutImprovement = 0;
    for (int epoch = 0; epoch < nEpochs; epoch++)
    {
        //Get generator seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        // Shuffle training data
        std::shuffle(trainInputs.begin(), trainInputs.end(), std::default_random_engine(seed));
        std::shuffle(trainOutputs.begin(), trainOutputs.end(), std::default_random_engine(seed));

        // Train on all samples
        for (int i = 0; i < static_cast<int>(trainInputs.size()); i++)
        {
            forwardPropagate(trainInputs[i]);
            backPropagate(trainOutputs[i]);
            updateWeightsAndBiases(learningRate);
        }

        // Calculate losses
        double trainLoss = calculateMSE(trainInputs, trainOutputs);
        double validLoss = calculateMSE(validInputs, validOutputs);

        // Check early stopping condition
        if (validLoss < bestValidLoss)
        {
            bestValidLoss = validLoss;
            epochsWithoutImprovement = 0;
        }
        else
        {
            epochsWithoutImprovement++;
            if (epochsWithoutImprovement >= patience)
            {
                std::cout << "Early stopping..." << std::endl;
                break;
            }
        }

        std::cout << "Epoch " << epoch << " Training MSE: " << trainLoss << ", Validation MSE: " << validLoss << std::endl;
    }
}

double NeuralNetwork::calculateMSE(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targetOutputs)
{
    double totalError = 0.0;
    for (int i = 0; i < static_cast<int>(inputs.size()); i++)
    {
        forwardPropagate(inputs[i]);
        for (int j = 0; j < static_cast<int>(targetOutputs[i].size()); j++)
        {
            double error = targetOutputs[i][j] - layers.back().neurons[j].output;
            totalError += error * error;
        }
    }
    return totalError / static_cast<int>(inputs.size());
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &inputs)
{
    forwardPropagate(inputs);
    std::vector<double> outputs;
    for (auto &neuron : layers.back().neurons)
    {
        outputs.push_back(neuron.output);
    }
    return outputs;
}

double NeuralNetwork::accuracy(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targetOutputs)
{
    int correctCount = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::vector<double> prediction = predict(inputs[i]);
        int predictedLabel = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        int actualLabel = std::distance(targetOutputs[i].begin(), std::max_element(targetOutputs[i].begin(), targetOutputs[i].end()));

        if (predictedLabel == actualLabel)
        {
            ++correctCount;
        }
    }

    return static_cast<double>(correctCount) / inputs.size();
}