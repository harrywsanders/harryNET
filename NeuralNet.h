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

class NeuralNetwork
{
public:
    std::vector<Layer> layers;

    void forwardPropagate(std::vector<double> &inputs) {}

    void backPropagate(std::vector<double> &targetOutputs) {}

    void updateWeightsAndBiases(double learningRate) {}

    void train(std::vector<std::vector<double>> &trainInputs, std::vector<std::vector<double>> &trainOutputs, std::vector<std::vector<double>> &validInputs, std::vector<std::vector<double>> &validOutputs, double learningRate, int nEpochs, int patience);

    double calculateMSE(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targetOutputs);

    std::vector<double> predict(std::vector<double> &inputs);
};