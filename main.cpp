#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include 'neuron.h'
#include 'layers.h'
#include 'NeuralNet.h'

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "neural_network.h" // Assuming the neural network classes are defined here.

int main()
{
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;

    std::ifstream file("mnist_train.csv");
    std::string line;
    while (getline(file, line))
    {
        std::vector<double> input;
        std::vector<double> output(10, 0.0);

        std::istringstream ss(line);
        std::string value;
        int index = 0;
        while (getline(ss, value, ','))
        {
            double num = std::stod(value);
            if (index == 0)
            {
                // Convert to one-hot encoding.
                output[(int)num] = 1.0;
            }
            else
            {
                // Normalize to the range 0-1.
                input.push_back(num / 255.0);
            }
            index++;
        }

        inputs.push_back(input);
        outputs.push_back(output);
    }

    // Assuming a simple structure with one hidden layer of 50 neurons.
    NeuralNetwork network;
    network.layers.push_back(Layer(784)); // Input
    network.layers.push_back(Layer(50));  // Hidden
    network.layers.push_back(Layer(10));  // Output

    // Train the network using arbitrary values.
    network.train(inputs, outputs, inputs, outputs, 0.01, 100, 10);

    double trainingAccuracy = network.accuracy(inputs, outputs);
    std::cout << "Training accuracy: " << trainingAccuracy << std::endl;

    return 0;
}
