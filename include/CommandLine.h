struct Options {
    std::string trainingDataPath;
    std::string testDataPath;
    int numEpochs;
    double learningRate;
    int batchSize;
    int patience;
    double lambda;
};

Options parseCommandLineArgs(int argc, char* argv[]) {
    // Set default values
    Options options;
    options.trainingDataPath = "train.csv";
    options.testDataPath = "test.csv";
    options.numEpochs = 10;
    options.learningRate = 0.01;
    options.patience = 10;
    options.batchSize = 32;
    options.lambda = 0.01;

    // Override with command line arguments
    if (argc > 1) options.trainingDataPath = argv[1];
    if (argc > 2) options.testDataPath = argv[2];
    if (argc > 3) options.numEpochs = std::stoi(argv[3]);
    if (argc > 4) options.learningRate = std::stod(argv[4]);
    if (argc > 5) options.patience = std::stoi(argv[5]);
    if (argc > 6) options.batchSize = std::stoi(argv[6]);
    if(argc > 7) options.lambda = std::stod(argv[7]);

    return options;
}
