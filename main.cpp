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

    void backPropagate() {
        // Implement backpropagation here
    }

    void updateWeightsAndBiases() {
        // Implement weight and bias updates here
    }
};
