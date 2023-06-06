/**
 * @class Layer
 *
 * @brief Represents a layer of neurons in a neural network.
 *
 * This class is essentially a container for a set of neurons. It provides a way to group neurons together into a layer, which simplifies the implementation of the neural network.
 *
 * Note: This class does not provide methods for computations on the layer (such as computing outputs of neurons in the layer or updating their weights). These computations are done in the NeuralNetwork class.
 */

class Layer
{
public:
    std::vector<Neuron> neurons;
    std::vector<double> inputs; 
    Layer(size_t nNeurons, size_t nInputsPerNeuron)
    {
        for (size_t i = 0; i < nNeurons; i++)
        {
            neurons.push_back(Neuron(nInputsPerNeuron));
        }
    }
};
