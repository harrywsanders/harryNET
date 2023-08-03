# Compiler
CXX = g++
# Flags
CXXFLAGS = -Wall -std=c++14
# Include directories for header files
INCLUDES = -I./include
# Libraries for linking (including Googletest)
LIBS = -lgtest -lgtest_main -pthread
# Source files directory
SRC_DIR = ./src
# Object files directory
OBJ_DIR = ./obj
# Binary directory
BIN_DIR = ./bin
# Test directory
TEST_DIR = .

# Target executable name
TARGET = $(BIN_DIR)/harryNET

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/tests.cpp)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TEST_TARGET = $(BIN_DIR)/tests

# Build rules
all: $(TARGET) run_tests

$(TARGET): $(OBJECTS)
	$(CXX) $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test build rule
tests: $(TEST_TARGET)
	./$(TEST_TARGET)   # Command to run the tests automatically after building

$(TEST_TARGET): $(TEST_OBJECTS) $(filter-out $(OBJ_DIR)/main.o, $(OBJECTS))
	$(CXX) $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/*