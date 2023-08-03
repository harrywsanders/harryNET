#include <gtest/gtest.h>
#include "../include/CommandLine.h"

TEST(CommandLineTest, DefaultValues) {
    int argc = 1;
    char* argv[] = {(char*)"program_name"};
    Options options = parseCommandLineArgs(argc, argv);

    EXPECT_EQ(options.trainingDataPath, "train.csv");
    EXPECT_EQ(options.testDataPath, "test.csv");
    EXPECT_EQ(options.numEpochs, 100);
    EXPECT_DOUBLE_EQ(options.learningRate, 0.01);
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


// Additional tests for other methods like backPropagate, train, predict, etc. can be added here
