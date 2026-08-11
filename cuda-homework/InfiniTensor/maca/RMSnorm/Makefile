MACA_PATH ?= /opt/maca
CUCC ?= $(MACA_PATH)/tools/cu-bridge/bin/cucc
TARGET := rmsnorm
SOURCES := main.cu rms_norm.cu
RMS_BLOCK_THREADS ?= 512
RMS_USE_SHARED_CACHE ?= 1
RMS_CACHE_HALF ?= 0
CUCC_FLAGS ?= -O3 -std=c++17 -DRMS_BLOCK_THREADS=$(RMS_BLOCK_THREADS) \
	-DRMS_USE_SHARED_CACHE=$(RMS_USE_SHARED_CACHE) -DRMS_CACHE_HALF=$(RMS_CACHE_HALF)

.PHONY: all test bench clean

all: $(TARGET)

$(TARGET): $(SOURCES) rms_norm.cuh
	MACA_PATH=$(MACA_PATH) CPATH=$(MACA_PATH)/tools/cu-bridge/include \
		$(CUCC) $(CUCC_FLAGS) $(SOURCES) -o $@

test: $(TARGET)
	./$(TARGET) test

bench: $(TARGET)
	./$(TARGET) bench 4096 4096 500 both

clean:
	rm -f $(TARGET)
