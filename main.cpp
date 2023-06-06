#include <vector>
#include <cmath>

class Neuron {
public:
    double weight, bias, output, delta;
};

class Layer {
public:
    std::vector<Neuron> neurons;
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;

    void forwardPropagate() {
        for (auto& layer : layers) {
            for (auto& neuron : layer.neurons) {
                double activation = neuron.weight * neuron.input + neuron.bias;
                neuron.output = 1 / (1 + std::exp(-activation)); // sigmoid activation function
            }
        }
    }

    void backPropagate(std::vector<double>& targetOutputs) {
        // Implement backpropagation here
    }

    void updateWeightsAndBiases() {
        // Implement weight and bias updates here
    }
};

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