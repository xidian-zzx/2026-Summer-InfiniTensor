NVCC ?= /usr/local/cuda/bin/nvcc
TARGET := kvcache
SOURCES := main.cu kv_cache.cu

ARCH_FLAGS ?= -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89
NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo $(ARCH_FLAGS)

.PHONY: all test bench clean

all: $(TARGET)

$(TARGET): $(SOURCES) kv_cache.cuh
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@

test: $(TARGET)
	./$(TARGET) test

bench: $(TARGET)
	./$(TARGET) bench 8 16 4096 8 128 1000 both

clean:
	rm -f $(TARGET)
