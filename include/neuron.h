/*
 * This class represents a single neuron in a neural network. Each neuron has the following properties:
 * - A set of weights, one for each input to the neuron. The weights are initialized using He initialization.
 * - A bias, which is also initialized using He initialization.
 * - An output value, which is computed during the forward propagation step.
 * - A delta value, which is computed during the backward propagation step.
 *
 * The neuron uses the sigmoid function as its activation function. However, this class does not include methods for computing the activation function or its derivative. These computations are performed in the NeuralNetwork class.
 *
 */

#include <cmath>
#include <vector>
#include <random>

class Neuron
{
public:
    Neuron(size_t nInputs)
    {
        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / nInputs));
        for (size_t i = 0; i < nInputs; i++)
        {
            weights.push_back(distribution(generator));
        }
        bias = distribution(generator);
    }
    std::vector<double> weights;
    double bias;
    double output;
    double delta;
};
