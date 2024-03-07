CXX=g++
CXXFLAGS=-I./include -std=c++17 -Wall -Wextra

# Directory for object files
OBJDIR=./obj

# Source and test source files
SOURCES=$(filter-out ./src/*tests.cpp, $(wildcard ./src/*.cpp))
TEST_SOURCES=$(wildcard ./src/*tests.cpp)

# Object files and test object files
OBJECTS=$(SOURCES:./src/%.cpp=$(OBJDIR)/%.o)
TEST_OBJECTS=$(TEST_SOURCES:./src/%.cpp=$(OBJDIR)/%.o)

# Executable and test executable names
EXECUTABLE=$(OBJDIR)/main
TEST_EXECUTABLE=$(OBJDIR)/tests

# Default target
main: $(EXECUTABLE)

# Link the executable
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile source files into object files
$(OBJDIR)/%.o: ./src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test target
test: $(TEST_EXECUTABLE)

# Link the test executable
$(TEST_EXECUTABLE): $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Rule to run tests
run_test: test
	./$(TEST_EXECUTABLE)

# Rule to run main application
run_main: all
	./$(EXECUTABLE)

# Clean build files
clean:
	rm -rf $(OBJDIR)

.PHONY: all clean test run_test run_all
