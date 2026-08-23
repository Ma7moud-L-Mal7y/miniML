# Compiler and Flags
CXX      := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -O2 -Iinclude

# Directories
SRC_DIR  := src
INC_DIR  := include
TEST_DIR := test
BUILD_DIR:= build
OBJ_DIR  := $(BUILD_DIR)/obj

# OS detection — decides which shell commands to use
ifeq ($(OS),Windows_NT)
    MKDIR   = if not exist "$(subst /,\,$(1))" mkdir "$(subst /,\,$(1))"
    RMDIR   = if exist "$(subst /,\,$(1))" rmdir /s /q "$(subst /,\,$(1))"
    EXE_EXT := .exe
else
    MKDIR   = mkdir -p "$(1)"
    RMDIR   = rm -rf "$(1)"
    EXE_EXT :=
endif

# Sources and Objects
SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Test Sources and Executables
TEST_SRCS:= $(wildcard $(TEST_DIR)/*.cpp)
TEST_BINS:= $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%$(EXE_EXT), $(TEST_SRCS))

ifeq ($(OS),Windows_NT)
WIN_TEST_BINS := $(subst /,\,$(TEST_BINS))
endif

# Default target
.PHONY: all clean test

all: $(TEST_BINS)

# Rule to compile library source files (.cpp -> .o)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(call MKDIR,$(OBJ_DIR))
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to link test binaries
$(BUILD_DIR)/%$(EXE_EXT): $(TEST_DIR)/%.cpp $(OBJS)
	@$(call MKDIR,$(BUILD_DIR))
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@

# Convenience command to run all built tests
ifeq ($(OS),Windows_NT)
test: all
	@echo ---------------------------------------
	@echo Running all tests in $(BUILD_DIR)...
	@echo ---------------------------------------
	@for %%T in ($(WIN_TEST_BINS)) do ( \
		echo Executing %%T... && \
		%%T || exit 1 && \
		echo --------------------------------------- \
	)
else
test: all
	@echo ---------------------------------------
	@echo Running all tests in $(BUILD_DIR)...
	@echo ---------------------------------------
	@for test_bin in $(TEST_BINS); do \
		echo Executing $$test_bin...; \
		$$test_bin || exit 1; \
		echo ---------------------------------------; \
	done
endif

# Clean built artifacts
clean:
	@$(call RMDIR,$(BUILD_DIR))
	@echo Cleaned build directory.