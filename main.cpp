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


void loadDataset(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs) {
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);

    while (getline(file, line)) {
        std::vector<double> input;
        std::vector<double> output(10, 0.0);

        std::istringstream ss(line);
        std::string value;
        int index = 0;
        while (getline(ss, value, ',')) {
            double num = std::stod(value);
            if (index == 0) {
                output[(int)num] = 1.0;
            } else {
                input.push_back(num / 255.0);
            }
            index++;
        }

        inputs.push_back(input);
        outputs.push_back(output);
    }
}

int main(int argc, char* argv[]) {
    if (argc > 6) {
        std::cout << "Usage: " << argv[0] << " [<training_data>] [<test_data>] [<num_epochs>] [<learning_rate>] [<patience>]" << std::endl;
        return 1;
    }

    Options options = parseCommandLineArgs(argc, argv);

    std::vector<std::vector<double>> trainingInputs, trainingOutputs;
    std::vector<std::vector<double>> testInputs, testOutputs;

    // Load datasets
    cout << "Loading datasets..." << endl;
    loadDataset(options.trainingDataPath, trainingInputs, trainingOutputs);
    loadDataset(options.testDataPath, testInputs, testOutputs);
    cout << "Done." << endl;

    // Initialize and train the network
    NeuralNetwork network;
    cout << "Initializing network..." << endl;
    network.layers.push_back(Layer(784, 784));
    network.layers.push_back(Layer(50, 784));
    network.layers.push_back(Layer(10, 50));
    cout << "Done." << endl;

    cout << "Training network..." << endl;
    network.train(trainingInputs, trainingOutputs, testInputs, testOutputs, options.learningRate, options.numEpochs, options.patience);
    cout << "Done." << endl;

    // Measure accuracy on the test set
    cout << "Measuring accuracy on test set..." << endl;
    double testAccuracy = network.accuracy(testInputs, testOutputs);
    std::cout << "Test accuracy: " << testAccuracy << std::endl;

    return 0;
}