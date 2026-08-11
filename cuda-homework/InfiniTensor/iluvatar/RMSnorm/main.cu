#include "rms_norm.cuh"

#include <algorithm>
#include <cmath>
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

// 下面两个转换函数只用于测试程序。它们把 host 端的 float/half 统一起来，
// 生成数据、算参考答案和比较误差就能共用同一套模板代码。
template<typename T>
float as_float(T value) {
  return static_cast<float>(value);
}

template<>
float as_float<half>(half value) {
  return __half2float(value);
}

template<typename T>
T as_type(float value) {
  return static_cast<T>(value);
}

template<>
half as_type<half>(float value) {
  return __float2half_rn(value);
}

template<typename T>
bool run_case(int rows, int hidden_dim, int iterations, bool print_perf) {
  const size_t element_count = static_cast<size_t>(rows) * hidden_dim;
  const size_t matrix_bytes = element_count * sizeof(T);
  const size_t weight_bytes = static_cast<size_t>(hidden_dim) * sizeof(T);
  std::vector<T> input(element_count);
  std::vector<T> weight(hidden_dim);
  std::vector<T> output(element_count);
  std::vector<float> reference(element_count);

  std::mt19937 generator(20260810);
  std::uniform_real_distribution<float> x_distribution(-1.0f, 1.0f);
  std::uniform_real_distribution<float> w_distribution(0.5f, 1.5f);
  // 固定随机种子后，每次运行的数据完全一致，误差和性能问题更容易复现。
  for (T& value : input) { value = as_type<T>(x_distribution(generator)); }
  for (T& value : weight) { value = as_type<T>(w_distribution(generator)); }

  constexpr float eps = 1.0e-5f;
  // CPU 参考实现故意写得很直白：每行先算平方和，再计算这一行的输出。
  // 平方和用 double，让参考结果尽量准确，方便判断 GPU 端的数值误差。
  for (int row = 0; row < rows; ++row) {
    double square_sum = 0.0;
    const size_t row_offset = static_cast<size_t>(row) * hidden_dim;
    for (int col = 0; col < hidden_dim; ++col) {
      const double x = as_float(input[row_offset + col]);
      square_sum += x * x;
    }
    const float inv_rms = 1.0f / std::sqrt(
        static_cast<float>(square_sum / hidden_dim) + eps);
    for (int col = 0; col < hidden_dim; ++col) {
      reference[row_offset + col] = as_float(input[row_offset + col])
                                    * inv_rms * as_float(weight[col]);
    }
  }

  T* device_input = nullptr;
  T* device_weight = nullptr;
  T* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, matrix_bytes));
  CUDA_CHECK(cudaMalloc(&device_weight, weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_output, matrix_bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), matrix_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_weight, weight.data(), weight_bytes, cudaMemcpyHostToDevice));

  // 先单独执行一次并拷回结果，完成正确性检查。
  CUDA_CHECK(launch_rms_norm(device_input, device_weight, device_output,
                             rows, hidden_dim, eps));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, matrix_bytes, cudaMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;
  for (size_t i = 0; i < element_count; ++i) {
    const float actual = as_float(output[i]);
    const float abs_error = std::abs(actual - reference[i]);
    const float rel_error = abs_error / std::max(std::abs(reference[i]), 1.0e-6f);
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
  }

  const float abs_tolerance = std::is_same<T, half>::value ? 2.0e-3f : 2.0e-5f;
  const bool passed = max_abs_error <= abs_tolerance;
  std::printf("%-5s rows=%-5d hidden=%-6d max_abs=%9.3e max_rel=%9.3e %s",
              std::is_same<T, half>::value ? "half" : "float", rows, hidden_dim,
              max_abs_error, max_rel_error, passed ? "PASS" : "FAIL");

  if (print_perf) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // 先预热 20 次，让 CUDA 上下文、缓存和 GPU 频率进入稳定状态。
    for (int i = 0; i < 20; ++i) {
      CUDA_CHECK(launch_rms_norm(device_input, device_weight, device_output,
                                 rows, hidden_dim, eps));
    }
    // CUDA event 只记录 GPU 时间，不把 CPU 提交 kernel 的耗时混进来。
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
      CUDA_CHECK(launch_rms_norm(device_input, device_weight, device_output,
                                 rows, hidden_dim, eps));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float us_per_launch = elapsed_ms * 1000.0f / iterations;
    // logical_BW 从算法角度统计 input + weight + output 的总字节数。
    // weight 往往会命中 L2，因此这个“有效带宽”可能高于物理显存带宽。
    const double logical_bytes = static_cast<double>(matrix_bytes) * 2.0
                                 + static_cast<double>(weight_bytes) * rows;
    const double effective_gbps = logical_bytes / (us_per_launch * 1.0e3);
    std::printf("  time=%8.3f us  logical_BW=%7.1f GB/s", us_per_launch, effective_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  std::printf("\n");

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_weight));
  CUDA_CHECK(cudaFree(device_output));
  return passed;
}

int run_correctness_suite() {
  struct Shape { int rows; int hidden; };
  // 这些尺寸覆盖标量兜底、warp 常见宽度、block 缓存和超宽回退路径。
  const Shape shapes[] = {
      {7, 127}, {13, 128}, {9, 768}, {5, 1000},
      {11, 1024}, {3, 4096}, {2, 4103}, {1, 16385}};
  bool all_passed = true;
  for (const Shape& shape : shapes) {
    all_passed &= run_case<float>(shape.rows, shape.hidden, 1, false);
    all_passed &= run_case<half>(shape.rows, shape.hidden, 1, false);
  }
  return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

void print_usage(const char* program) {
  std::printf("Usage:\n");
  std::printf("  %s test\n", program);
  std::printf("  %s bench [rows] [hidden_dim] [iterations] [float|half|both]\n", program);
}

int main(int argc, char** argv) {
  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  std::printf("Device: %s (compat %d.%d, warp=%d)\n", property.name,
              property.major, property.minor, property.warpSize);

  if (argc == 1 || std::string(argv[1]) == "test") {
    return run_correctness_suite();
  }
  if (std::string(argv[1]) != "bench") {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const int rows = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int hidden_dim = argc > 3 ? std::atoi(argv[3]) : 4096;
  const int iterations = argc > 4 ? std::atoi(argv[4]) : 200;
  const std::string dtype = argc > 5 ? argv[5] : "both";
  if (rows <= 0 || hidden_dim <= 0 || iterations <= 0
      || (dtype != "float" && dtype != "half" && dtype != "both")) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  bool passed = true;
  if (dtype == "float" || dtype == "both") {
    passed &= run_case<float>(rows, hidden_dim, iterations, true);
  }
  if (dtype == "half" || dtype == "both") {
    passed &= run_case<half>(rows, hidden_dim, iterations, true);
  }
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
