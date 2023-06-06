struct Options {
    std::string trainingDataPath;
    std::string testDataPath;
    int numEpochs;
    double learningRate;
    int patience;
};

Options parseCommandLineArgs(int argc, char* argv[]) {
    // Set default values
    Options options;
    options.trainingDataPath = "train.csv";
    options.testDataPath = "test.csv";
    options.numEpochs = 100;
    options.learningRate = 0.01;
    options.patience = 10;

    // Override with command line arguments
    if (argc > 1) options.trainingDataPath = argv[1];
    if (argc > 2) options.testDataPath = argv[2];
    if (argc > 3) options.numEpochs = std::stoi(argv[3]);
    if (argc > 4) options.learningRate = std::stod(argv[4]);
    if (argc > 5) options.patience = std::stoi(argv[5]);

    return options;
}

void printProgressBar(int current, int total)
{
    int percent = (current * 100) / total;
    std::cout << "\rTraining progress: [";

    int i = 0;
    for (; i < percent/2; i++) std::cout << "=";

    for (; i < 50; i++) std::cout << " ";

    std::cout << "] " << percent << "%   ";

    std::cout.flush(); 
}