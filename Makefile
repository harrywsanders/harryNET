CXX = g++
CXXFLAGS = -std=c++14 -O2 -Wall -Wextra

# The build target executable
TARGET = neural_net.exe

# Source files
SRC = main.cpp

# Header files
HEADERS = neuron.h layer.h NeuralNet.h CommandLine.h

# Object files
OBJ = $(SRC:.cpp=.o)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $<

clean:
	$(RM) $(TARGET) $(OBJ)
