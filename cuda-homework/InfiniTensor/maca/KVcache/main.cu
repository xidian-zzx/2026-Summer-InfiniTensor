#include "kv_cache.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(error));                                   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

template<typename T>
T make_value(float value) {
  return static_cast<T>(value);
}

template<>
half make_value<half>(float value) {
  return __float2half_rn(value);
}

template<typename T>
bool same_bits(T lhs, T rhs) {
  // KV Cache 只搬运数据、不做数学运算，所以理论上应逐 bit 相同，无需误差范围。
  return lhs == rhs;
}

template<>
bool same_bits<half>(half lhs, half rhs) {
  // MACA 把 __half_as_ushort 做成了 device-only，host 端直接调会编译失败。
  // 这里把两个字节原样拷进整数再比较，检查力度仍然是逐 bit 的。
  std::uint16_t lhs_bits = 0;
  std::uint16_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  return lhs_bits == rhs_bits;
}

template<typename T>
struct DeviceBuffers {
  // 测试程序用一个简单的 RAII 容器集中管理显存。run_case 退出时析构函数
  // 自动释放所有指针，避免中途 return 后漏掉 cudaFree。
  T* new_k = nullptr;
  T* new_v = nullptr;
  T* cache_k = nullptr;
  T* cache_v = nullptr;
  T* out_k = nullptr;
  T* out_v = nullptr;
  int* positions = nullptr;

  ~DeviceBuffers() {
    cudaFree(new_k);
    cudaFree(new_v);
    cudaFree(cache_k);
    cudaFree(cache_v);
    cudaFree(out_k);
    cudaFree(out_v);
    cudaFree(positions);
  }
};

template<typename T>
bool run_correctness_case(int batch_size, int tokens, int max_seq_len,
                          int kv_heads, int head_dim) {
  // 一次用例会验证两件事：update 是否只改了指定位置，以及 gather 能否
  // 按倒序 positions 把刚才写入的 K/V 完整读回来。
  const int elements_per_token = kv_heads * head_dim;
  const size_t compact_elements =
      static_cast<size_t>(batch_size) * tokens * elements_per_token;
  const size_t cache_elements =
      static_cast<size_t>(batch_size) * max_seq_len * elements_per_token;
  const size_t compact_bytes = compact_elements * sizeof(T);
  const size_t cache_bytes = cache_elements * sizeof(T);
  std::vector<T> new_k(compact_elements);
  std::vector<T> new_v(compact_elements);
  std::vector<T> cache_k(cache_elements, make_value<T>(-7.0f));
  std::vector<T> cache_v(cache_elements, make_value<T>(-9.0f));
  std::vector<T> expected_k = cache_k;
  std::vector<T> expected_v = cache_v;
  std::vector<T> gathered_k(compact_elements);
  std::vector<T> gathered_v(compact_elements);
  std::vector<int> update_positions(batch_size * tokens);
  std::vector<int> gather_positions(batch_size * tokens);

  std::mt19937 generator(20260810);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (T& value : new_k) { value = make_value<T>(distribution(generator)); }
  for (T& value : new_v) { value = make_value<T>(distribution(generator)); }

  // 每个 batch 内使用互不重复的位置。
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < tokens; ++token) {
      update_positions[batch * tokens + token] = (batch + token * 2) % max_seq_len;
    }
  }
  // 等 update_positions 全部填好后再构造倒序列表，顺便验证乱序读取。
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < tokens; ++token) {
      gather_positions[batch * tokens + token] =
          update_positions[batch * tokens + (tokens - 1 - token)];
    }
  }

  // CPU 端构造 update 之后应该得到的完整缓存。
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < tokens; ++token) {
      const int position = update_positions[batch * tokens + token];
      const size_t source =
          (static_cast<size_t>(batch) * tokens + token) * elements_per_token;
      const size_t destination =
          (static_cast<size_t>(batch) * max_seq_len + position) * elements_per_token;
      std::copy_n(new_k.data() + source, elements_per_token,
                  expected_k.data() + destination);
      std::copy_n(new_v.data() + source, elements_per_token,
                  expected_v.data() + destination);
    }
  }

  DeviceBuffers<T> device;
  CUDA_CHECK(cudaMalloc(&device.new_k, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.new_v, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.cache_k, cache_bytes));
  CUDA_CHECK(cudaMalloc(&device.cache_v, cache_bytes));
  CUDA_CHECK(cudaMalloc(&device.out_k, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.out_v, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.positions,
                        update_positions.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(device.new_k, new_k.data(), compact_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.new_v, new_v.data(), compact_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.cache_k, cache_k.data(), cache_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.cache_v, cache_v.data(), cache_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.positions, update_positions.data(),
                        update_positions.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 第一阶段：执行 GPU update，再把整个缓存拷回 CPU。检查整个缓存可以确认
  // 指定位置写对了，也能确认其他位置没有被误改。
  CUDA_CHECK(launch_kv_cache_update(
      device.new_k, device.new_v, device.positions, device.cache_k, device.cache_v,
      batch_size, tokens, max_seq_len, kv_heads, head_dim));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(cache_k.data(), device.cache_k, cache_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cache_v.data(), device.cache_v, cache_bytes,
                        cudaMemcpyDeviceToHost));

  size_t update_errors = 0;
  for (size_t i = 0; i < cache_elements; ++i) {
    update_errors += !same_bits(cache_k[i], expected_k[i]);
    update_errors += !same_bits(cache_v[i], expected_v[i]);
  }

  // 第二阶段：把 positions 换成倒序列表，验证 gather 的重排能力。
  CUDA_CHECK(cudaMemcpy(device.positions, gather_positions.data(),
                        gather_positions.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(launch_kv_cache_gather(
      device.cache_k, device.cache_v, device.positions, device.out_k, device.out_v,
      batch_size, tokens, max_seq_len, kv_heads, head_dim));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gathered_k.data(), device.out_k, compact_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gathered_v.data(), device.out_v, compact_bytes,
                        cudaMemcpyDeviceToHost));

  size_t gather_errors = 0;
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < tokens; ++token) {
      const int source_token = tokens - 1 - token;
      const size_t expected_source =
          (static_cast<size_t>(batch) * tokens + source_token) * elements_per_token;
      const size_t output_offset =
          (static_cast<size_t>(batch) * tokens + token) * elements_per_token;
      for (int element = 0; element < elements_per_token; ++element) {
        gather_errors += !same_bits(gathered_k[output_offset + element],
                                    new_k[expected_source + element]);
        gather_errors += !same_bits(gathered_v[output_offset + element],
                                    new_v[expected_source + element]);
      }
    }
  }

  const bool passed = update_errors == 0 && gather_errors == 0;
  std::printf("%-5s B=%-2d T=%-3d Smax=%-4d H=%-3d D=%-3d "
              "update_errors=%zu gather_errors=%zu %s\n",
              std::is_same<T, half>::value ? "half" : "float",
              batch_size, tokens, max_seq_len, kv_heads, head_dim,
              update_errors, gather_errors, passed ? "PASS" : "FAIL");
  return passed;
}

template<typename T>
bool run_benchmark(int batch_size, int tokens, int max_seq_len,
                   int kv_heads, int head_dim, int iterations) {
  // benchmark 只搬运本次涉及的 token；大块 cache 只负责提供真实的地址跨度，
  // 不会在每次迭代前后整体清零或复制。
  const int elements_per_token = kv_heads * head_dim;
  const size_t compact_elements =
      static_cast<size_t>(batch_size) * tokens * elements_per_token;
  const size_t cache_elements =
      static_cast<size_t>(batch_size) * max_seq_len * elements_per_token;
  const size_t compact_bytes = compact_elements * sizeof(T);
  const size_t cache_bytes = cache_elements * sizeof(T);
  std::vector<T> new_k(compact_elements);
  std::vector<T> new_v(compact_elements);
  std::vector<T> gathered_k(compact_elements);
  std::vector<T> gathered_v(compact_elements);
  std::vector<int> positions(batch_size * tokens);
  std::mt19937 generator(20260810);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (T& value : new_k) { value = make_value<T>(distribution(generator)); }
  for (T& value : new_v) { value = make_value<T>(distribution(generator)); }
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < tokens; ++token) {
      positions[batch * tokens + token] = token;
    }
  }

  DeviceBuffers<T> device;
  CUDA_CHECK(cudaMalloc(&device.new_k, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.new_v, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.cache_k, cache_bytes));
  CUDA_CHECK(cudaMalloc(&device.cache_v, cache_bytes));
  CUDA_CHECK(cudaMalloc(&device.out_k, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.out_v, compact_bytes));
  CUDA_CHECK(cudaMalloc(&device.positions, positions.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(device.new_k, new_k.data(), compact_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.new_v, new_v.data(), compact_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.positions, positions.data(),
                        positions.size() * sizeof(int), cudaMemcpyHostToDevice));

  // update 和 gather 各预热 20 次，让 CUDA 上下文和 GPU 频率先稳定下来。
  for (int i = 0; i < 20; ++i) {
    CUDA_CHECK(launch_kv_cache_update(
        device.new_k, device.new_v, device.positions, device.cache_k, device.cache_v,
        batch_size, tokens, max_seq_len, kv_heads, head_dim));
    CUDA_CHECK(launch_kv_cache_gather(
        device.cache_k, device.cache_v, device.positions, device.out_k, device.out_v,
        batch_size, tokens, max_seq_len, kv_heads, head_dim));
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // 两个操作分别计时，方便看出连续写与按位置读取的差异。
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(launch_kv_cache_update(
        device.new_k, device.new_v, device.positions, device.cache_k, device.cache_v,
        batch_size, tokens, max_seq_len, kv_heads, head_dim));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float update_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&update_ms, start, stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(launch_kv_cache_gather(
        device.cache_k, device.cache_v, device.positions, device.out_k, device.out_v,
        batch_size, tokens, max_seq_len, kv_heads, head_dim));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float gather_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gather_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaMemcpy(gathered_k.data(), device.out_k, compact_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gathered_v.data(), device.out_v, compact_bytes,
                        cudaMemcpyDeviceToHost));
  size_t errors = 0;
  for (size_t i = 0; i < compact_elements; ++i) {
    errors += !same_bits(gathered_k[i], new_k[i]);
    errors += !same_bits(gathered_v[i], new_v[i]);
  }

  // 每次操作都要处理 K 和 V，各有一次读、一次写，因此逻辑数据量是
  // compact_bytes * 4。这里报告的是有效带宽，便于不同形状横向比较。
  const double logical_bytes = static_cast<double>(compact_bytes) * 4.0;
  const double update_us = update_ms * 1000.0 / iterations;
  const double gather_us = gather_ms * 1000.0 / iterations;
  std::printf("%-5s B=%d T=%d Smax=%d H=%d D=%d  "
              "update=%7.3f us (%7.1f GB/s)  gather=%7.3f us (%7.1f GB/s)  %s\n",
              std::is_same<T, half>::value ? "half" : "float",
              batch_size, tokens, max_seq_len, kv_heads, head_dim,
              update_us, logical_bytes / (update_us * 1.0e3),
              gather_us, logical_bytes / (gather_us * 1.0e3),
              errors == 0 ? "PASS" : "FAIL");
  return errors == 0;
}

int run_tests() {
  struct Shape { int batch; int tokens; int max_seq; int heads; int dim; };
  const Shape shapes[] = {
      {2, 1, 17, 3, 7},       // decode 的标量回退路径
      {5, 1, 33, 8, 128},     // decode 的 16 字节向量路径
      {2, 3, 17, 3, 7},       // 标量回退路径
      {3, 5, 64, 8, 128},     // 常见 half8/float4 路径
      {1, 4, 33, 2, 64}};
  bool passed = true;
  for (const Shape& shape : shapes) {
    passed &= run_correctness_case<float>(shape.batch, shape.tokens,
                                          shape.max_seq, shape.heads, shape.dim);
    passed &= run_correctness_case<half>(shape.batch, shape.tokens,
                                         shape.max_seq, shape.heads, shape.dim);
  }
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

void print_usage(const char* program) {
  std::printf("Usage:\n");
  std::printf("  %s test\n", program);
  std::printf("  %s bench [B] [new_tokens] [max_seq] [kv_heads] [head_dim] "
              "[iterations] [float|half|both]\n", program);
}

int main(int argc, char** argv) {
  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  std::printf("Device: %s (CUDA bridge capability %d.%d)\n",
              property.name, property.major, property.minor);
  if (argc == 1 || std::string(argv[1]) == "test") { return run_tests(); }
  if (std::string(argv[1]) != "bench") {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const int batch = argc > 2 ? std::atoi(argv[2]) : 8;
  const int tokens = argc > 3 ? std::atoi(argv[3]) : 16;
  const int max_seq = argc > 4 ? std::atoi(argv[4]) : 4096;
  const int heads = argc > 5 ? std::atoi(argv[5]) : 8;
  const int dim = argc > 6 ? std::atoi(argv[6]) : 128;
  const int iterations = argc > 7 ? std::atoi(argv[7]) : 1000;
  const std::string dtype = argc > 8 ? argv[8] : "both";
  if (batch <= 0 || tokens <= 0 || tokens > max_seq || max_seq <= 0
      || heads <= 0 || dim <= 0 || iterations <= 0
      || (dtype != "float" && dtype != "half" && dtype != "both")) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  bool passed = true;
  if (dtype == "float" || dtype == "both") {
    passed &= run_benchmark<float>(batch, tokens, max_seq, heads, dim, iterations);
  }
  if (dtype == "half" || dtype == "both") {
    passed &= run_benchmark<half>(batch, tokens, max_seq, heads, dim, iterations);
  }
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
