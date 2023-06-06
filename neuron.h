/**
 * @class Neuron
 *
 * @brief Represents a neuron in a neural network.
 *
 * Each neuron in the network has a set of weights (one for each input), a bias, an output value (computed in the forward propagation step), and a delta value (computed in the backward propagation step).
 * The neuron uses the sigmoid function as its activation function.
 * 
 * Note: This class does not include a method for computing the activation function or its derivative. These computations are done in the NeuralNetwork class.
 */

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
    std::vector<double> weights; 
    double bias;                
    double output;               
    double delta;                
};