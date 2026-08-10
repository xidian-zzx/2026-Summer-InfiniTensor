CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp
LDLIBS ?= -lOpenCL -fopenmp
TARGET := build/gemv_opencl

.PHONY: all run quick clean

all: $(TARGET)

$(TARGET): src/main.cpp kernels/gemv_q8.cl
	mkdir -p build
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

quick: $(TARGET)
	./$(TARGET) --quick

clean:
	rm -f $(TARGET)
