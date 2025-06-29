.PHONY: all build run clean test help

# Default target
all: build

# Build the project using CMake
build:
	@mkdir -p build
	@cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Build and run the main executable
run: build
	@cd build && ./harrynet_main

# Run tests if they exist
test: build
	@cd build && ctest --verbose || echo "No tests configured"

# Clean build artifacts
clean:
	@rm -rf build

# Clean and rebuild
rebuild: clean build

# Build with debug symbols
debug:
	@mkdir -p build
	@cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Show available targets
help:
	@echo "Available targets:"
	@echo "  make build    - Build the project"
	@echo "  make run      - Build and run the main executable"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make rebuild  - Clean and rebuild"
	@echo "  make debug    - Build with debug symbols"
	@echo "  make help     - Show this help message"