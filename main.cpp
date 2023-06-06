#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "neuron.h"
#include "layers.h"
#include "NeuralNet.h"
#include "CommandLine.h"

void loadDataset(const std::string &filename, std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs)
{
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);

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
                output[(int)num] = 1.0;
            }
            else
            {
                input.push_back(num / 255.0);
            }
            index++;
        }

        inputs.push_back(input);
        outputs.push_back(output);
    }
}

int main(int argc, char *argv[])
{
    if (argc > 8)
    {
        std::cout << "Usage: " << argv[0] << " [<training_data>] [<test_data>] [<num_epochs>] [<learning_rate>] [<patience>] [<momentum>] [<epsilon>]" << std::endl;
        return 1;
    }

    Options options = parseCommandLineArgs(argc, argv);

    std::vector<std::vector<double>> trainingInputs, trainingOutputs;
    std::vector<std::vector<double>> testInputs, testOutputs;

    // Load datasets
    std::cout << "Loading datasets..." << std::endl;
    loadDataset(options.trainingDataPath, trainingInputs, trainingOutputs);
    loadDataset(options.testDataPath, testInputs, testOutputs);
    std::cout << "Done." << std::endl;

    // Initialize and train the network
    NeuralNetwork network;
    std::cout << "Initializing network..." << std::endl;
    network.layers.push_back(Layer(784, 784));
    network.layers.push_back(Layer(50, 784));
    network.layers.push_back(Layer(10, 50));
    std::cout << "Done." << std::endl;

    std::cout << "Training network..." << std::endl;
    network.train(trainingInputs, trainingOutputs, testInputs, testOutputs, options.learningRate, options.numEpochs, options.patience, options.momentum, options.epsilon);
    std::cout << "Done." << std::endl;

    // Measure accuracy on the test set
    std::cout << "Measuring accuracy on test set..." << std::endl;
    double testAccuracy = network.accuracy(testInputs, testOutputs);
    std::cout << "Test accuracy: " << testAccuracy << std::endl;

    return 0;
}