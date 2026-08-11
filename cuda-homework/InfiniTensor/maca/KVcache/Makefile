MACA_PATH ?= /opt/maca
CUCC ?= $(MACA_PATH)/tools/cu-bridge/bin/cucc
TARGET := kvcache
SOURCES := main.cu kv_cache.cu

# 这些值是起点，不是写死的魔法数字。需要重新扫参数时，可以在 make
# 命令行覆盖，例如：make KV_DECODE_WARPS=4 KV_BULK_THREADS=512。
KV_DECODE_WARPS ?= 8
KV_BULK_THREADS ?= 512
CUCC_FLAGS ?= -O3 -std=c++17 -DKV_DECODE_WARPS=$(KV_DECODE_WARPS) \
	-DKV_BULK_THREADS=$(KV_BULK_THREADS)

.PHONY: all test bench clean

all: $(TARGET)

$(TARGET): $(SOURCES) kv_cache.cuh
	MACA_PATH=$(MACA_PATH) CPATH=$(MACA_PATH)/tools/cu-bridge/include \
		$(CUCC) $(CUCC_FLAGS) $(SOURCES) -o $@

test: $(TARGET)
	./$(TARGET) test

bench: $(TARGET)
	./$(TARGET) bench 8 16 4096 8 128 5000 both

clean:
	rm -f $(TARGET)
