# harryNET

> Hey there! Welcome to harryNET, a simple neural network built in C++! I created this network with the goal of using it as an educational tool on the base function of neural networks, and am continuing to update it with new features. 

## What's harryNET?

harryNET is a straightforward implementation of a neural network that allows you to build and train your own models. It's lightweight, simple, and constantly improving through updates!

## Usage

1. Clone the repository: `git clone https://github.com/your_username/harryNET.git`
2. Navigate to the project directory: `cd harryNET`
3. Compile the code: `make`
4. Run the program: `./neural_net.exe <training_data> <test_data> [<num_epochs>] [<learning_rate>] [<patience>]`

> ### Command-line Arguments
> 
> - `<training_data>`: Path to the training data file in CSV format.
> - `<test_data>`: Path to the test data file in CSV format.
> - `<num_epochs>` (optional): Number of epochs for training. Default: 100.
> - `<learning_rate>` (optional): Learning rate for weight updates. Default: 0.01.
> - `<patience>` (optional): Patience parameter for early stopping. Default: 10.

> ### Example Usage
> 
> ```
> ./neural_net.exe mnist_train.csv mnist_test.csv 200 0.001 20
> ```

## Goals for the Future

> - **Optimization Techniques**: Explore advanced optimization techniques like momentum, adaptive learning rate, or weight decay to improve the network's performance.
> - **Diverse Activation Functions**: Incorporate different activation functions such as ReLU, tanh, or softmax to expand the network's capabilities and experiment with their impact on learning.
> - **Regularization Methods**: Implement techniques like dropout or L1/L2 regularization to enhance generalization and prevent overfitting.
> - **Convolutional Neural Networks**: Extend the network architecture to support convolutional layers and pooling layers, enabling the handling of image-based datasets.
> - **Hyperparameter Tuning**: Develop a systematic hyperparameter tuning mechanism, like grid search or random search, to find optimal values for parameters such as learning rate, number of hidden layers, or number of neurons per layer.
> - **Additional Datasets**: Test the network's performance on various datasets to evaluate its ability to generalize across different problem domains.
> - **Model Persistence**: Implement functionality to save and load trained models, allowing easy reuse and transferability across different sessions or applications.
> - **Visualization**: Create visualization tools to analyze the network's learning progress, such as plotting loss curves or visualizing learned weights.
