# Compiler
CXX = g++
# Flags
CXXFLAGS = -Wall -std=c++14 -g -O1 -Wextra
CXXFLAGS_PROF = -Wall -std=c++14 -g -O1 -Wextra 
# Include directories for header files
INCLUDES = -I./include -I/users/harrysanders/googletest/googletest/include -I/opt/homebrew/opt/libomp/include
# Libraries for linking 
LIBS = -L/users/harrysanders/googletest/build/lib -L/opt/homebrew/opt/gperftools/lib -L/opt/homebrew/opt/libomp/lib -lgtest -lgtest_main -pthread -lprofiler -lomp
# Source files directory
SRC_DIR = ./src
# Object files directory
OBJ_DIR = ./obj
# Binary directory
BIN_DIR = ./bin
# Test directory
TEST_DIR = ./src

# Target executable name
TARGET = $(BIN_DIR)/neural_net.exe

# Main source files (excluding tests)
MAIN_SOURCES = $(wildcard $(SRC_DIR)/main.cpp)
MAIN_OBJECTS = $(MAIN_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/tests.cpp)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TEST_TARGET = $(BIN_DIR)/tests

# Build rules
all: main test

profiling: CXXFLAGS = $(CXXFLAGS_PROF)
profiling: clean main

main: $(MAIN_OBJECTS)
	$(CXX) $^ -o $(TARGET) $(LIBS)

run_main: main
	./$(TARGET) mnist_train.csv mnist_test.csv

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test build rule
test: $(TEST_TARGET)
	./$(TEST_TARGET)  

$(TEST_TARGET): $(TEST_OBJECTS) $(filter-out $(OBJ_DIR)/main.o, $(MAIN_OBJECTS))
	$(CXX) $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/*
