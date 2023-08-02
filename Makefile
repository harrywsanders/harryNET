CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra

# Source and object files directory
SRCDIR = src
OBJDIR = obj
BINDIR = bin
INCDIR = include


# The build target executable
TARGET = neural_net.exe
TARGET_PATH = $(BINDIR)/$(TARGET)

# Source and object files
SRC = $(wildcard $(SRCDIR)/*.cpp)
OBJ = $(SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
DEP = $(OBJ:.o=.d)

# Release and Debug build settings
RELEASE_FLAGS = -O2
DEBUG_FLAGS = -g -O0 -DDEBUG

# Default build
all: setup release

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: setup $(TARGET_PATH)

# Release build
release: CXXFLAGS += $(RELEASE_FLAGS)
release: setup $(TARGET_PATH)

tests: $(OBJDIR)/tests.o
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/tests $(OBJDIR)/tests.o


# The target binary
$(TARGET_PATH): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation of source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(INCDIR)/*.h
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -MMD -c $< -o $@

# Include dependencies
-include $(DEP)

# Set up the directories
setup:
	mkdir -p $(SRCDIR)
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

# Clean up
clean:
	$(RM) -r $(BINDIR)
	$(RM) -r $(OBJDIR)

.PHONY: all setup clean debug release

# Additional flags for linking with Googletest
GTEST_FLAGS = -lgtest -lgtest_main -pthread

# Test target executable
TEST_TARGET = tests
TEST_TARGET_PATH = $(BINDIR)/$(TEST_TARGET)

# Test source and object files
TEST_SRC = tests.cpp
TEST_OBJ = $(OBJDIR)/tests.o

# Test target
test: setup $(TEST_TARGET_PATH)

# The test binary
$(TEST_TARGET_PATH): $(TEST_OBJ) $(OBJ)
	$(CXX) $(CXXFLAGS) $(GTEST_FLAGS) -I$(INCDIR) -o $@ $^

# Compilation of test source file into object file
$(OBJDIR)/tests.o: $(TEST_SRC) $(INCDIR)/*.h
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -MMD -c $< -o $@
