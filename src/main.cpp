#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../include/NeuralNet.h"
#include "../include/CommandLine.h"


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
    if (argc > 8) {
        std::cout << "Usage: " << argv[0] << " [<training_data>] [<test_data>] [<num_epochs>] [<learning_rate>] [<patience>] [<batch_size>] [<l2 lambda>]" << std::endl;
        return 1;
    }

    Options options = parseCommandLineArgs(argc, argv);

    std::vector<std::vector<double>> trainInputsRaw, trainOutputsRaw;
    std::vector<std::vector<double>> testInputsRaw, testOutputsRaw;

    loadDataset(options.trainingDataPath, trainInputsRaw, trainOutputsRaw);
    loadDataset(options.testDataPath, testInputsRaw, testOutputsRaw);

    std::cout << "Data has been loaded successfully." << std::endl;

    std::vector<Eigen::VectorXd> trainInputs, trainOutputs;
    std::vector<Eigen::VectorXd> testInputs, testOutputs;

    for (const auto &v : trainInputsRaw) {
        trainInputs.push_back(Eigen::Map<const Eigen::VectorXd>(v.data(), v.size()));
    }

    for (const auto &v : trainOutputsRaw) {
        trainOutputs.push_back(Eigen::Map<const Eigen::VectorXd>(v.data(), v.size()));
    }

    for (const auto &v : testInputsRaw) {
        testInputs.push_back(Eigen::Map<const Eigen::VectorXd>(v.data(), v.size()));
    }

    for (const auto &v : testOutputsRaw) {
        testOutputs.push_back(Eigen::Map<const Eigen::VectorXd>(v.data(), v.size()));
    }

    // Create neural network
    NeuralNetwork nn;
    // Initialize each layer
    size_t nInputs = 784;
    size_t nNeuronsHidden1 = 512;
    size_t nNeuronsHidden2 = 256;
    size_t nNeuronsOutput = 10; 
    Layer inputLayer(nInputs, nInputs); 
    Layer hiddenLayer1(nNeuronsHidden1, nInputs); 
    Layer hiddenLayer2(nNeuronsHidden2, nNeuronsHidden1); 
    Layer outputLayer(nNeuronsOutput, nNeuronsHidden2);

    nn.layers.push_back(inputLayer);
    nn.layers.push_back(hiddenLayer1);
    nn.layers.push_back(hiddenLayer2);
    nn.layers.push_back(outputLayer);
    std::cout << "Network initialized. Training beginning." << std::endl;


    // Train the network
    nn.train(trainInputs, trainOutputs, testInputs, testOutputs, options.learningRate, options.numEpochs, options.batchSize, options.patience,options.lambda);

    // Evaluate accuracy on test data
    double accuracy = nn.accuracy(testInputs, testOutputs);
    std::cout << "Test accuracy: " << accuracy << std::endl;

    return 0;
}
