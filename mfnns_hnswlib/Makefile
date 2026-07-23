CXX ?= g++
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/mfnns_hnsw_tool
SOURCE := src/mfnns_hnsw_tool.cpp
TABLE5_TARGET := $(BUILD_DIR)/table5_dataset_runner
TABLE5_SOURCE := table5_reproduction/table5_dataset_runner.cpp
HEADERS := $(wildcard src/hnswlib/*.h)

CXXFLAGS ?= -O3 -DNDEBUG
REQUIRED_CXXFLAGS := -std=c++17 -fopenmp -march=native -Wall -Wextra -Wpedantic
LDLIBS += -pthread

.PHONY: all clean table5 tool

all: $(TARGET) $(TABLE5_TARGET)

tool: $(TARGET)

table5: $(TABLE5_TARGET)

$(TARGET): $(SOURCE) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(REQUIRED_CXXFLAGS) -isystem src $(SOURCE) -o $(TARGET) $(LDLIBS)

$(TABLE5_TARGET): $(TABLE5_SOURCE) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(REQUIRED_CXXFLAGS) -isystem src $(TABLE5_SOURCE) -o $(TABLE5_TARGET) $(LDLIBS)

clean:
	@if [ -z "$(BUILD_DIR)" ] || [ "$(BUILD_DIR)" = "/" ] || [ "$(BUILD_DIR)" = "$(HOME)" ]; then \
		echo "Refusing unsafe BUILD_DIR=$(BUILD_DIR)" >&2; exit 1; \
	fi
	rm -f $(TARGET) $(TABLE5_TARGET)
