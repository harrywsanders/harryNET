#include <gtest/gtest.h>
#include "../include/CommandLine.h"

TEST(CommandLineTest, DefaultValues) {
    int argc = 1;
    char* argv[] = {(char*)"program_name"};
    Options options = parseCommandLineArgs(argc, argv);

    EXPECT_EQ(options.trainingDataPath, "train.csv");
    EXPECT_EQ(options.testDataPath, "test.csv");
    EXPECT_EQ(options.numEpochs, 100);
    EXPECT_DOUBLE_EQ(options.learningRate, 0.001);
    EXPECT_EQ(options.patience, 10);
    EXPECT_EQ(options.batchSize, 32);
    EXPECT_DOUBLE_EQ(options.lambda, 0.01);
}

TEST(CommandLineTest, CustomValues) {
    int argc = 8;
    char* argv[] = {(char*)"program_name", (char*)"custom_train.csv", (char*)"custom_test.csv", (char*)"200", (char*)"0.05", (char*)"15", (char*)"64", (char*)"0.02"};
    Options options = parseCommandLineArgs(argc, argv);

    EXPECT_EQ(options.trainingDataPath, "custom_train.csv");
    EXPECT_EQ(options.testDataPath, "custom_test.csv");
    EXPECT_EQ(options.numEpochs, 200);
    EXPECT_DOUBLE_EQ(options.learningRate, 0.05);
    EXPECT_EQ(options.patience, 15);
    EXPECT_EQ(options.batchSize, 64);
    EXPECT_DOUBLE_EQ(options.lambda, 0.02);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "../include/NeuralNet.h"

// Test for NeuralNetwork forward propagation
TEST(NeuralNetworkTest, ForwardPropagation) {
    // Create Neural Network
    NeuralNetwork nn;
    
    // Create and add input layer
    Layer inputLayer(3, 3, LayerType::Dense);
    nn.layers.push_back(inputLayer);

    // Create and add hidden layer
    Layer hiddenLayer(4, 3, LayerType::Dense);
    nn.layers.push_back(hiddenLayer);

    // Create and add output layer
    Layer outputLayer(2, 4, LayerType::Dense);
    nn.layers.push_back(outputLayer);

    // Create inputs
    Eigen::VectorXd inputs(3);
    inputs << 1.0, 2.0, 3.0;

    // Forward Propagate
    nn.forwardPropagate(inputs);

}

TEST(NeuralNetworkTest, BackPropagation) {
    NeuralNetwork nn;

    // Create and add layers
    Layer inputLayer(3, 3, LayerType::Dense);
    Layer hiddenLayer(4, 3, LayerType::Dense);
    Layer outputLayer(2, 4, LayerType::Dense);

    // Example weights and biases for hidden layer
    hiddenLayer.weights << 0.1, 0.2, 0.3,
                           0.4, 0.5, 0.6,
                           0.7, 0.8, 0.9,
                           1.0, 1.1, 1.2;
    hiddenLayer.bias << 0.5, 0.5, 0.5, 0.5;

    // Example weights and biases for output layer
    outputLayer.weights << 0.1, 0.2, 0.3, 0.4,
                           0.5, 0.6, 0.7, 0.8;
    outputLayer.bias << 0.5, 0.5;

    nn.layers.push_back(inputLayer);
    nn.layers.push_back(hiddenLayer);
    nn.layers.push_back(outputLayer);

    Eigen::VectorXd inputs(3);
    inputs << 1.0, 2.0, 3.0;
    nn.forwardPropagate(inputs);

    Eigen::VectorXd targetOutputs(2);
    targetOutputs << 0.5, 0.5;
    nn.backPropagate(targetOutputs);

    // Calculate expected values manually
    Eigen::VectorXd expectedHiddenOutput(4);
    expectedHiddenOutput << std::max(0.5 + 1.4, 0.0), 
                            std::max(0.5 + 3.2, 0.0), 
                            std::max(0.5 + 5.0, 0.0), 
                            std::max(0.5 + 6.8, 0.0);

    // Add more expectations based on the actual values of weights and biases
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(nn.layers[1].output[i], expectedHiddenOutput[i], 1e-5);
    }
}


TEST(NeuralNetworkTest, Training) {
    NeuralNetwork nn;
    // Create and add layers
    Layer inputLayer(3, 3, LayerType::Dense);
    Layer hiddenLayer(4, 3, LayerType::Dense);
    Layer outputLayer(2, 4, LayerType::Dense);
    nn.layers.push_back(inputLayer);
    nn.layers.push_back(hiddenLayer);
    nn.layers.push_back(outputLayer);

    std::vector<Eigen::VectorXd> trainInputs, trainOutputs, validInputs, validOutputs;
    // Populate inputs and outputs

    trainInputs.push_back(Eigen::VectorXd(3));
    trainInputs[0] << 1.0, 2.0, 3.0;
    trainOutputs.push_back(Eigen::VectorXd(2));
    trainOutputs[0] << 0.5, 0.5;

    validInputs.push_back(Eigen::VectorXd(3));
    validInputs[0] << 1.0, 2.0, 3.0;
    validOutputs.push_back(Eigen::VectorXd(2));
    validOutputs[0] << 0.5, 0.5;

    double learningRate = 0.01;
    int nEpochs = 100;
    int batchSize = 32;
    int patience = 10;
    double lambda = 0.01;

    nn.train(trainInputs, trainOutputs, validInputs, validOutputs, learningRate, nEpochs, batchSize, patience, lambda, false);
}