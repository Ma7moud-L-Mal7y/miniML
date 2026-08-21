# Compiler and Flags
CXX      := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -O2 -Iinclude

# Directories
SRC_DIR  := src
INC_DIR  := include
TEST_DIR := test
BUILD_DIR:= build
OBJ_DIR  := $(BUILD_DIR)/obj

# Sources and Objects
SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Test Sources and Executables
TEST_SRCS:= $(wildcard $(TEST_DIR)/*.cpp)
TEST_BINS:= $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%, $(TEST_SRCS))

# Default target
.PHONY: all clean test

all: $(TEST_BINS)

# Rule to compile library source files (.cpp -> .o)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to link test binaries
$(BUILD_DIR)/%: $(TEST_DIR)/%.cpp $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@

# Convenience command to run all built tests
test: all
	@echo "---------------------------------------"
	@echo "Running all tests in $(BUILD_DIR)..."
	@echo "---------------------------------------"
	@for test_bin in $(TEST_BINS); do \
		echo "Executing $$test_bin..."; \
		./$$test_bin || exit 1; \
		echo "---------------------------------------"; \
	done

# Clean built artifacts
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory."