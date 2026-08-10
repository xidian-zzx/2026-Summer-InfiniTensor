NVCC ?= /usr/local/cuda/bin/nvcc
TARGET := rmsnorm
SOURCES := main.cu rms_norm.cu

# Build one binary for the local RTX 3080 (sm_86) and the target RTX 4090 (sm_89).
ARCH_FLAGS ?= -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89
NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo $(ARCH_FLAGS)

.PHONY: all test bench clean

all: $(TARGET)

$(TARGET): $(SOURCES) rms_norm.cuh
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@

test: $(TARGET)
	./$(TARGET) test

bench: $(TARGET)
	./$(TARGET) bench 4096 4096 200 both

clean:
	rm -f $(TARGET)
