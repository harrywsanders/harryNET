#include <vector>
#include <cmath>
#include <random>
#include <iostream>

class Neuron
{
public:
    Neuron(size_t nInputs)
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (size_t i = 0; i < nInputs; i++)
        {
            weights.push_back(distribution(generator));
        }
        bias = distribution(generator);
    }
    std::vector<double> weights; // weights for each input
    double bias;                 // bias value
    double output;               // output value
    double delta;                // delta value for backpropagation
};

class Layer
{
public:
    std::vector<Neuron> neurons;
    Layer(size_t nNeurons, size_t nInputsPerNeuron) {
        for(size_t i = 0; i < nNeurons; i++) {
            neurons.push_back(Neuron(nInputsPerNeuron));
        }
    }
};

class NeuralNetwork
{
public:
    std::vector<Layer> layers;

    void forwardPropagate(std::vector<double> &inputs) {}

    void backPropagate(std::vector<double> &targetOutputs) {}

    void updateWeightsAndBiases(double learningRate) {}

    void train(std::vector<std::vector<double>> &trainInputs, std::vector<std::vector<double>> &trainOutputs, std::vector<std::vector<double>> &validInputs, std::vector<std::vector<double>> &validOutputs, double learningRate, int nEpochs);

    double calculateMSE(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targetOutputs);

    std::vector<double> predict(std::vector<double> &inputs);


};

void NeuralNetwork::forwardPropagate(std::vector<double> &inputs)
{
    for (auto &layer : layers)
    {
        std::vector<double> outputs;
        for (auto &neuron : layer.neurons)
        {
            double activation = neuron.bias;
            for (int i = 0; i < neuron.weights.size(); i++)
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
    for (int i = 0; i < outputLayer.neurons.size(); i++)
    {
        double output = outputLayer.neurons[i].output;
        double target = targetOutputs[i];
        double error = target - output;
        outputLayer.neurons[i].delta = error * output * (1 - output); // derivative of MSE loss with respect to output * derivative of sigmoid
    }

    // Calculate hidden layer deltas
    for (int l = layers.size() - 2; l >= 0; l--) {
        Layer &hiddenLayer = layers[l];
        Layer &nextLayer = layers[l+1];
        for (int i = 0; i < hiddenLayer.neurons.size(); i++)
        {
            double output = hiddenLayer.neurons[i].output;
            double error = 0.0;
            for (int j = 0; j < nextLayer.neurons.size(); j++)
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
            for (int i = 0; i < neuron.weights.size(); i++)
            {
                neuron.weights[i] += learningRate * neuron.delta * inputs[i]; // update weight
            }
            neuron.bias += learningRate * neuron.delta; // update bias
        }
    }
}

void NeuralNetwork::train(std::vector<std::vector<double>> &trainInputs, std::vector<std::vector<double>> &trainOutputs, std::vector<std::vector<double>> &validInputs, std::vector<std::vector<double>> &validOutputs, double learningRate, int nEpochs)
{
    for (int epoch = 0; epoch < nEpochs; epoch++) 
    {
        for (int i = 0; i < trainInputs.size(); i++) 
        {
            forwardPropagate(trainInputs[i]);
            backPropagate(trainOutputs[i]);
            updateWeightsAndBiases(learningRate);
        }
        double trainLoss = calculateMSE(trainInputs, trainOutputs);
        double validLoss = calculateMSE(validInputs, validOutputs);
        std::cout << "Epoch " << epoch << " Training MSE: " << trainLoss << ", Validation MSE: " << validLoss << std::endl;
    }
}

double NeuralNetwork::calculateMSE(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targetOutputs)
{
    double totalError = 0.0;
    for (int i = 0; i < inputs.size(); i++)
    {
        forwardPropagate(inputs[i]);
        for (int j = 0; j < targetOutputs[i].size(); j++)
        {
            double error = targetOutputs[i][j] - layers.back().neurons[j].output;
            totalError += error * error;
        }
    }
    return totalError / inputs.size();
}

std::vector<double> NeuralNetwork::predict(std::vector<double> &inputs)
{
    forwardPropagate(inputs);
    std::vector<double> outputs;
    for (auto &neuron : layers.back().neurons)
    {
        outputs.push_back(neuron.output);
    }
    return outputs;
}