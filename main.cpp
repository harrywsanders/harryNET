#include <vector>
#include <cmath>

class Neuron {
public:
    std::vector<double> weights; // weights for each input
    double bias; // bias value
    double output; // output value
    double delta; // delta value for backpropagation
};

class Layer {
public:
    std::vector<Neuron> neurons;
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;

        void forwardPropagate(std::vector<double>& inputs) {}

    void backPropagate(std::vector<double>& targetOutputs) {}

    void updateWeightsAndBiases(double learningRate) {}
};

void NeuralNetwork::forwardPropagate(std::vector<double>& inputs){
    for (auto& layer : layers) {
            std::vector<double> outputs;
            for (auto& neuron : layer.neurons) {
                double activation = neuron.bias;
                for (int i = 0; i < neuron.weights.size(); i++) {
                    activation += neuron.weights[i] * inputs[i];
                }
                neuron.output = 1 / (1 + std::exp(-activation)); // sigmoid activation function
                outputs.push_back(neuron.output);
            }
            inputs = outputs; // outputs of this layer are inputs to the next layer
        }
}

void NeuralNetwork::backPropagate(std::vector<double>& targetOutputs) {
    // Calculate output layer deltas
    Layer& outputLayer = layers.back();
    for (int i = 0; i < outputLayer.neurons.size(); i++) {
        double output = outputLayer.neurons[i].output;
        double target = targetOutputs[i];
        double error = target - output;
        outputLayer.neurons[i].delta = error * output * (1 - output); // derivative of MSE loss with respect to output * derivative of sigmoid
    }

    // Calculate hidden layer deltas
    Layer& hiddenLayer = layers[0]; // assuming single hidden layer
    Layer& nextLayer = layers[1];
    for (int i = 0; i < hiddenLayer.neurons.size(); i++) {
        double output = hiddenLayer.neurons[i].output;
        double error = 0.0;
        for (int j = 0; j < nextLayer.neurons.size(); j++) {
            error += nextLayer.neurons[j].delta * nextLayer.neurons[j].weights[i]; // weights from hidden layer to output layer
        }
        hiddenLayer.neurons[i].delta = error * output * (1 - output); // derivative of sigmoid
    }
}

void NeuralNetwork::updateWeightsAndBiases(double learningRate) {
     std::vector<double> inputs;
        if (layers.size() > 1) {
            for (auto& neuron : layers[layers.size() - 2].neurons) {
                inputs.push_back(neuron.output);
            }
        }

        for (auto& layer : layers) {
            for (auto& neuron : layer.neurons) {
                for (int i = 0; i < neuron.weights.size(); i++) {
                    neuron.weights[i] += learningRate * neuron.delta * inputs[i]; // update weight
                }
                neuron.bias += learningRate * neuron.delta; // update bias
            }
        }
    }
