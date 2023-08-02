
#include <gtest/gtest.h>
#include <chrono>
#include "include/CommandLine.h"
#include "include/NeuralNet.h"

// Speed test for parsing command-line arguments
TEST(SpeedTest, ParseCommandLineArgs) {
    int argc = 3;
    char* argv[] = {(char*)"program", (char*)"train.csv", (char*)"test.csv"};
    auto start = std::chrono::high_resolution_clock::now();
    auto options = parseCommandLineArgs(argc, argv);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken to parse command-line arguments: " << duration.count() << " microseconds" << std::endl;
}

// Additional speed tests can be written for the methods in NeuralNet.h and layers.h

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
