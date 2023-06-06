#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include 'neuron.h'
#include 'layers.h'
#include 'NeuralNet.h'


void loadDataset(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs) {
    std::ifstream file(filename);
    std::string line;
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
    // Set default values
    std::string trainingDataPath = "train.csv";
    std::string testDataPath = "test.csv";
    int numEpochs = 100;
    double learningRate = 0.01;
    int patience = 10;

    // Override with command line arguments
    if (argc > 1) {
        trainingDataPath = argv[1];
    }
    if (argc > 2) {
        testDataPath = argv[2];
    }
    if (argc > 3) {
        numEpochs = std::stoi(argv[3]);
    }
    if (argc > 4) {
        learningRate = std::stod(argv[4]);
    }
    if (argc > 5) {
        patience = std::stoi(argv[5]);
    }

    if (argc > 6) {
        std::cout << "Usage: " << argv[0] << " [<training_data>] [<test_data>] [<num_epochs>] [<learning_rate>] [<patience>]" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> trainingInputs, trainingOutputs;
    std::vector<std::vector<double>> testInputs, testOutputs;

    // Load datasets
    loadDataset(trainingDataPath, trainingInputs, trainingOutputs);
    loadDataset(testDataPath, testInputs, testOutputs);

    // Initialize and train the network
    NeuralNetwork network;
    network.layers.push_back(Layer(784));
    network.layers.push_back(Layer(50));
    network.layers.push_back(Layer(10));

    network.train(trainingInputs, trainingOutputs, testInputs, testOutputs, learningRate, numEpochs, patience);

    // Measure accuracy on the test set
    double testAccuracy = network.accuracy(testInputs, testOutputs);
    std::cout << "Test accuracy: " << testAccuracy << std::endl;

    return 0;
}


