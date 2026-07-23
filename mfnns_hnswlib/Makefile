CXX ?= g++
BUILD_DIR ?= /tmp/mfnns_hnswlib_build
TARGET := $(BUILD_DIR)/mfnns_hnsw_tool
SOURCE := src/mfnns_hnsw_tool.cpp
HEADERS := $(wildcard src/hnswlib/*.h)

CXXFLAGS ?= -O3 -DNDEBUG
REQUIRED_CXXFLAGS := -std=c++17 -fopenmp -march=native -Wall -Wextra -Wpedantic
LDLIBS += -pthread

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(REQUIRED_CXXFLAGS) -isystem src $(SOURCE) -o $(TARGET) $(LDLIBS)

clean:
	@if [ -z "$(BUILD_DIR)" ] || [ "$(BUILD_DIR)" = "/" ] || [ "$(BUILD_DIR)" = "$(HOME)" ]; then \
		echo "Refusing unsafe BUILD_DIR=$(BUILD_DIR)" >&2; exit 1; \
	fi
	rm -f $(TARGET)
