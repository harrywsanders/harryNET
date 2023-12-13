#include <gtest/gtest.h>
#include "../include/CommandLine.h"

TEST(CommandLineTest, DefaultValues) {
    int argc = 1;
    char* argv[] = {(char*)"program_name"};
    Options options = parseCommandLineArgs(argc, argv);

    EXPECT_EQ(options.trainingDataPath, "train.csv");
    EXPECT_EQ(options.testDataPath, "test.csv");
    EXPECT_EQ(options.numEpochs, 10);
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

    // Expected values for hidden layer output
    Eigen::VectorXd expectedHiddenOutput(4);
    expectedHiddenOutput << 1.9, 3.7, 5.5, 7.3;

    Eigen::VectorXd expectedOutputLayerOutput(2);
    expectedOutputLayerOutput << 6.0, 13.36;

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(nn.layers[1].output[i], expectedHiddenOutput[i], 1e-5);
    }

    // Check the output of output layer
    for (int i = 0; i < 2; i++) {
        EXPECT_NEAR(nn.layers[2].output[i], expectedOutputLayerOutput[i], 1e-5);
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

    nn.train(trainInputs, trainOutputs, validInputs, validOutputs, learningRate, nEpochs, batchSize, patience, lambda, false, false);
}

//Add NN training test with different layer activation functions

TEST(NeuralNetworkTest, SigmoidActivation) {
    NeuralNetwork nn;
    // Create and add layers
    std::unique_ptr<ActivationFunction> sigmoid(new Sigmoid());
    Layer inputLayer(3, 3, LayerType::Dense, std::move(sigmoid));
    Layer hiddenLayer(4, 3, LayerType::Dense, std::move(sigmoid));
    Layer outputLayer(2, 4, LayerType::Dense, std::move(sigmoid));
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

    nn.train(trainInputs, trainOutputs, validInputs, validOutputs, learningRate, nEpochs, batchSize, patience, lambda, false, false);
}

TEST(NeuralNetworkTest, Tanh) {
    NeuralNetwork nn;
    // Create and add layers
    std::unique_ptr<ActivationFunction> tanh(new Tanh());
    Layer inputLayer(3, 3, LayerType::Dense, std::move(tanh));
    Layer hiddenLayer(4, 3, LayerType::Dense, std::move(tanh));
    Layer outputLayer(2, 4, LayerType::Dense, std::move(tanh));
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

    nn.train(trainInputs, trainOutputs, validInputs, validOutputs, learningRate, nEpochs, batchSize, patience, lambda, false, false);
}

#include "../include/activations.h"

TEST(ActivationTest, Sigmoid) {
    Sigmoid sigmoid;
    EXPECT_DOUBLE_EQ(sigmoid.compute(0.0), 0.5);
    EXPECT_DOUBLE_EQ(sigmoid.compute(1.0), 0.7310585786300049);
    EXPECT_DOUBLE_EQ(sigmoid.compute(-1.0), 0.2689414213699951);
    EXPECT_DOUBLE_EQ(sigmoid.derivative(0.0), 0.25);
    EXPECT_DOUBLE_EQ(sigmoid.derivative(1.0), 0.19661193324148185);
    EXPECT_DOUBLE_EQ(sigmoid.derivative(-1.0), 0.19661193324148185);
}

TEST(ActivationTest, Tanh) {
    Tanh tanh;
    EXPECT_DOUBLE_EQ(tanh.compute(0.0), 0.0);
    EXPECT_DOUBLE_EQ(tanh.compute(1.0), 0.7615941559557649);
    EXPECT_DOUBLE_EQ(tanh.compute(-1.0), -0.7615941559557649);
    EXPECT_DOUBLE_EQ(tanh.derivative(0.0), 1.0);
    EXPECT_DOUBLE_EQ(tanh.derivative(1.0), 0.41997434161402614);
    EXPECT_DOUBLE_EQ(tanh.derivative(-1.0), 0.41997434161402614);
}

TEST(ActivationTest, ReLU) {
    ReLU relu;
    EXPECT_DOUBLE_EQ(relu.compute(0.0), 0.0);
    EXPECT_DOUBLE_EQ(relu.compute(1.0), 1.0);
    EXPECT_DOUBLE_EQ(relu.compute(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(relu.derivative(0.0), 0.0);
    EXPECT_DOUBLE_EQ(relu.derivative(1.0), 1.0);
    EXPECT_DOUBLE_EQ(relu.derivative(-1.0), 0.0);
}

TEST(ActivationTest, Softmax) {
    Softmax softmax;
    Eigen::VectorXd input(3);
    input << 1.0, 2.0, 3.0;
    Eigen::VectorXd output = softmax.eigenCompute(input);
    EXPECT_DOUBLE_EQ(output[0], 0.09003057317038046);
    EXPECT_DOUBLE_EQ(output[1], 0.24472847105479764);
    EXPECT_DOUBLE_EQ(output[2], 0.6652409557748219);
}