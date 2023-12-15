#include <vector>
#include <cmath>
#include <random>
#include <utility>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../include/NeuralNet.h"
#include "../include/CommandLine.h"
#include "../include/activations.h"

void loadDataset(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs) {
    std::ifstream file(filename);
    if (!file) {
        // Handle error or throw an exception
        std::cout << "Error opening file: " << filename << std::endl;
        return;
    }

    // Estimate dataset size
    unsigned int numLines = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
    file.clear();
    file.seekg(0, std::ios::beg); 

    std::string line;
    std::getline(file, line);
    unsigned int numColumns = std::count(line.begin(), line.end(), ',') + 1;

    // Reserve capacity for inputs and outputs
    inputs.reserve(numLines);
    outputs.reserve(numLines);

    double reciprocal_255 = 1.0 / 255.0;
    std::string value;
    double num;
    while (std::getline(file, line)) {
        std::vector<double> input;
        input.reserve(numColumns - 1); 
        std::vector<double> output(10, 0.0); 

        std::stringstream ss(line);
        int index = 0;
        while (std::getline(ss, value, ',')) {
            num = std::stod(value);
            if (index == 0) {
                output[static_cast<int>(num)] = 1.0;
            } else {
                input.push_back(num * reciprocal_255);
            }
            index++;
        }

        inputs.push_back(std::move(input)); 
        outputs.push_back(std::move(output));
    }
    std::random_device rd;
    std::mt19937 g(rd());

    for (size_t i = 0; i < inputs.size(); ++i) {
        size_t j = std::uniform_int_distribution<size_t>(0, i)(g);
        std::swap(inputs[i], inputs[j]);
        std::swap(outputs[i], outputs[j]);
    }
}




int main(int argc, char* argv[]) {
    if (argc > 9) {
        std::cout << "Usage: " << argv[0] << " [<training_data>] [<test_data>] [<num_epochs>] [<learning_rate>] [<patience>] [<batch_size>] [<l2 lambda>] [<testing?>]" << std::endl;
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

    if (options.isTestMode) {
        // Reduce dataset size to 5%
        auto reduceDatasetSize = [](auto& data) {
            size_t newSize = static_cast<size_t>(data.size() * 0.02);
            data.resize(newSize);
        };
        reduceDatasetSize(trainInputs);
        reduceDatasetSize(trainOutputs);
        reduceDatasetSize(testInputs);
        reduceDatasetSize(testOutputs);

        std::cout << "Running in test mode with reduced dataset size." << std::endl;
    }

    // Create neural network
    NeuralNetwork nn;
    // Initialize each layer
    size_t nInputs = 784;
    size_t nNeuronsHidden1 = 512;
    size_t nNeuronsHidden2 = 256;
    size_t nNeuronsOutput = 10; 
    Layer inputLayer(nInputs, nInputs, std::make_unique<inputActivation>(), LayerType::Dense);
    Layer hiddenLayer1(nNeuronsHidden1, nInputs, std::make_unique<Sigmoid>(), LayerType::Dense);
    Layer hiddenLayer2(nNeuronsHidden2, nNeuronsHidden1, std::make_unique<Sigmoid>(), LayerType::Dense);
    Layer outputLayer(nNeuronsOutput, nNeuronsHidden2, std::make_unique<Softmax>(), LayerType::Dense);

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
