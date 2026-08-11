COREX_PATH ?= /usr/local/corex-4.4.0
CXX := $(COREX_PATH)/bin/clang++
TARGET := kvcache
SOURCES := main.cu kv_cache.cu
ARCH ?= ivcore11
KV_DECODE_WARPS ?= 8
KV_BULK_THREADS ?= 256
CXX_FLAGS ?= -x ivcore --cuda-path=$(COREX_PATH) --cuda-gpu-arch=$(ARCH) \
	-O3 -std=c++17 -Wno-format -DKV_DECODE_WARPS=$(KV_DECODE_WARPS) \
	-DKV_BULK_THREADS=$(KV_BULK_THREADS)
LINK_FLAGS ?= -L$(COREX_PATH)/lib64 -lcudart \
	-Wl,-rpath,$(COREX_PATH)/lib64

.PHONY: all test bench clean

all: $(TARGET)

$(TARGET): $(SOURCES) kv_cache.cuh
	$(CXX) $(CXX_FLAGS) $(SOURCES) $(LINK_FLAGS) -o $@

test: $(TARGET)
	LD_LIBRARY_PATH=$(COREX_PATH)/lib64 ./$(TARGET) test

bench: $(TARGET)
	LD_LIBRARY_PATH=$(COREX_PATH)/lib64 ./$(TARGET) bench 8 16 4096 8 128 1000 both

clean:
	rm -f $(TARGET)
