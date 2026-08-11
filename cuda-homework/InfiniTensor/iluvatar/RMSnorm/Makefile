COREX_PATH ?= /usr/local/corex-4.4.0
CXX := $(COREX_PATH)/bin/clang++
TARGET := rmsnorm
SOURCES := main.cu rms_norm.cu
ARCH ?= ivcore11
RMS_BLOCK_THREADS ?= 256
RMS_USE_SHARED_CACHE ?= 1
RMS_CACHE_HALF ?= 0
CXX_FLAGS ?= -x ivcore --cuda-path=$(COREX_PATH) --cuda-gpu-arch=$(ARCH) \
	-O3 -std=c++17 -Wno-format -DRMS_BLOCK_THREADS=$(RMS_BLOCK_THREADS) \
	-DRMS_USE_SHARED_CACHE=$(RMS_USE_SHARED_CACHE) -DRMS_CACHE_HALF=$(RMS_CACHE_HALF)
LINK_FLAGS ?= -L$(COREX_PATH)/lib64 -lcudart \
	-Wl,-rpath,$(COREX_PATH)/lib64

.PHONY: all test bench clean

all: $(TARGET)

$(TARGET): $(SOURCES) rms_norm.cuh
	$(CXX) $(CXX_FLAGS) $(SOURCES) $(LINK_FLAGS) -o $@

test: $(TARGET)
	LD_LIBRARY_PATH=$(COREX_PATH)/lib64 ./$(TARGET) test

bench: $(TARGET)
	LD_LIBRARY_PATH=$(COREX_PATH)/lib64 ./$(TARGET) bench 4096 4096 500 both

clean:
	rm -f $(TARGET)
