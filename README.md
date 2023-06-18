# harryNET

> Hey there! Welcome to harryNET, a simple neural network built in C++! I created this network to work through and deepen my understanding of neural networks, and to have a platform to constantly improve as my knowledge grows.

## What's harryNET?

harryNET is a straightforward implementation of a neural network that allows you to build and train your own models. It's lightweight, simple, and constantly improving through updates! Currently, it's running at around 98% accuracy on the MNIST classification task using the default parameters.

## Usage

1. Clone the repository: `git clone https://github.com/your_username/harryNET.git`
2. Navigate to the project directory: `cd harryNET`
3. Compile the code: `make`
4. Run the program: `./bin/neural_net.exe [<training_data>] [<test_data>] [<num_epochs>] [<learning_rate>] [<batch_size>] [<patience>]`

> ### Command-line Arguments
> 
> - `<training_data>`: Path to the training data file in CSV format.
> - `<test_data>`: Path to the test data file in CSV format.
> - `<num_epochs>` (optional): Number of epochs for training. Default: 100.
> - `<learning_rate>` (optional): Learning rate for weight updates. Default: 0.01.
> - `<batch_size>` (optional): Parameter for batching. Default: 32.
> - `<patience>` (optional): Patience parameter for early stopping. Default: 10.
> - `<l2 lambda>` (optional): Lambda parameter for l2 regularization. Controls for overfitting. Default: 0.01.

> ### Example Usage
> 
> ```
> ./bin/neural_net.exe mnist_train.csv mnist_test.csv 200 0.001 20 10 0.1
> ```

## Goals for the Future [in order of ambition]
- [x] **Model Persistence**: Implement functionality to save and load trained models, allowing easy reuse and transferability across different sessions or applications.
- [ ] **Hyperparameter Tuning**: Develop a systematic hyperparameter tuning mechanism, like grid search or random search, to find optimal values for parameters such as learning rate, number of hidden layers, or number of neurons per layer.
- [ ] **Diverse Activation Functions**: Incorporate different activation functions, like  ReLU, tanh, or softmax to expand the network's capabilities.
- [x] **Regularization Methods**: Implement a technique like L1/L2 regularization to enhance generalization and prevent overfitting.
- [x] **Optimization Techniques**: Explore advanced optimization techniques like momentum, adaptive learning rate, or weight decay to improve the network's performance.
- [ ] **GPU Acceleration**: Implement GPU usage through CUDA or otherwise to speed up the performance.
- [ ] **Convolutional Neural Networks**: Extend the network architecture to support convolutional layers and pooling layers.

