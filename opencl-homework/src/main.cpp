#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBlockSize = 32;
constexpr float kQMax = 127.0f;
constexpr float kMinScale = 1.0e-6f;
constexpr cl_int kPlatformNotFound = -1001;

struct Options {
    int rows = 122753;
    int cols = 2304;
    int local_size = 32;
    int warmup = 10;
    int iterations = 100;
    float alpha = 0.5f;
    std::string kernel_path;
};

void check(cl_int error, const char *operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed, OpenCL error " + std::to_string(error));
    }
}

// 这两个小函数只负责 fp32/fp16 的位转换，省得项目再背一个很大的第三方头文件。
uint16_t floatToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    uint32_t mantissa = bits & 0x007fffffU;
    int exponent = static_cast<int>((bits >> 23U) & 0xffU) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x00800000U) >> static_cast<unsigned>(1 - exponent);
        if (mantissa & 0x00001000U) mantissa += 0x00002000U;
        return static_cast<uint16_t>(sign | (mantissa >> 13U));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    if (mantissa & 0x00001000U) {
        mantissa += 0x00002000U;
        if (mantissa & 0x00800000U) {
            mantissa = 0;
            ++exponent;
            if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa >> 13U));
}

float halfToFloat(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x03ffU;
            bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint32_t mixBits(uint32_t x) {
    // 用下标直接造随机数，多线程时每一行也能稳定复现。
    x ^= x >> 16U;
    x *= 0x7feb352dU;
    x ^= x >> 15U;
    x *= 0x846ca68bU;
    return x ^ (x >> 16U);
}

float deterministicHalf(size_t index, uint32_t seed) {
    const uint32_t bits = mixBits(static_cast<uint32_t>(index) ^ seed);
    const float unit = static_cast<float>(bits >> 8U) * (1.0f / 16777216.0f);
    return halfToFloat(floatToHalf(unit - 0.5f));
}

int8_t quantize(float value, float scale) {
    const int rounded = static_cast<int>(std::nearbyint(value / scale));
    return static_cast<int8_t>(std::max(-127, std::min(127, rounded)));
}

struct DataSet {
    std::vector<int8_t> weight_q;
    std::vector<uint16_t> weight_scale;
    std::vector<uint16_t> activation;
    std::vector<int8_t> activation_q;
    std::vector<uint16_t> activation_scale;
    std::vector<float> reference_weight_q;
    std::vector<float> reference_both_q;
    std::vector<float> reference_original;
};

DataSet makeData(int rows, int cols, float alpha) {
    const int blocks = cols / kBlockSize;
    DataSet data;
    data.weight_q.resize(static_cast<size_t>(rows) * cols);
    data.weight_scale.resize(static_cast<size_t>(rows) * blocks);
    data.activation.resize(cols);
    data.activation_q.resize(cols);
    data.activation_scale.resize(blocks);
    data.reference_weight_q.resize(rows);
    data.reference_both_q.resize(rows);
    data.reference_original.resize(rows);

    std::vector<float> activation_f32(cols);
    for (int col = 0; col < cols; ++col) {
        activation_f32[col] = deterministicHalf(static_cast<size_t>(col), 66U);
        data.activation[col] = floatToHalf(activation_f32[col]);
    }
    for (int block = 0; block < blocks; ++block) {
        float max_abs = 0.0f;
        for (int lane = 0; lane < kBlockSize; ++lane) {
            max_abs = std::max(max_abs, std::fabs(activation_f32[block * kBlockSize + lane]));
        }
        const uint16_t scale_bits = floatToHalf(std::max(max_abs / kQMax, kMinScale));
        const float scale = halfToFloat(scale_bits);
        data.activation_scale[block] = scale_bits;
        for (int lane = 0; lane < kBlockSize; ++lane) {
            const int col = block * kBlockSize + lane;
            data.activation_q[col] = quantize(activation_f32[col], scale);
        }
    }

    // 每行互不依赖，数据生成、量化和 CPU 参考值干脆一起算掉。
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        float original_sum = 0.0f;
        float weight_q_sum = 0.0f;
        float both_q_sum = 0.0f;
        for (int block = 0; block < blocks; ++block) {
            float values[kBlockSize];
            float max_abs = 0.0f;
            const size_t base = static_cast<size_t>(row) * cols + block * kBlockSize;
            for (int lane = 0; lane < kBlockSize; ++lane) {
                values[lane] = deterministicHalf(base + lane, 2026U);
                max_abs = std::max(max_abs, std::fabs(values[lane]));
                original_sum += values[lane] * activation_f32[block * kBlockSize + lane];
            }
            const uint16_t scale_bits = floatToHalf(std::max(max_abs / kQMax, kMinScale));
            const float scale_a = halfToFloat(scale_bits);
            const float scale_b = halfToFloat(data.activation_scale[block]);
            data.weight_scale[static_cast<size_t>(row) * blocks + block] = scale_bits;
            int integer_dot = 0;
            float fp16_dot = 0.0f;
            for (int lane = 0; lane < kBlockSize; ++lane) {
                const int col = block * kBlockSize + lane;
                const int8_t q = quantize(values[lane], scale_a);
                data.weight_q[base + lane] = q;
                fp16_dot += static_cast<float>(q) * activation_f32[col];
                integer_dot += static_cast<int>(q) * static_cast<int>(data.activation_q[col]);
            }
            weight_q_sum += fp16_dot * scale_a;
            both_q_sum += static_cast<float>(integer_dot) * scale_a * scale_b;
        }
        data.reference_original[row] = alpha * original_sum;
        data.reference_weight_q[row] = alpha * weight_q_sum;
        data.reference_both_q[row] = alpha * both_q_sum;
    }
    return data;
}

std::string readTextFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

std::pair<std::string, std::string> findKernel(const Options &options, const char *argv0) {
    std::vector<std::string> candidates;
    if (!options.kernel_path.empty()) candidates.push_back(options.kernel_path);
    candidates.emplace_back("kernels/gemv_q8.cl");
    candidates.emplace_back("../kernels/gemv_q8.cl");
    const std::string executable = argv0 ? argv0 : "";
    const size_t slash = executable.find_last_of("/\\");
    if (slash != std::string::npos) {
        candidates.push_back(executable.substr(0, slash + 1) + "../kernels/gemv_q8.cl");
    }
    for (const auto &path : candidates) {
        std::string source = readTextFile(path);
        if (!source.empty()) return {path, std::move(source)};
    }
    throw std::runtime_error("找不到 kernels/gemv_q8.cl，请在 opencl-homework 目录运行，或用 --kernel 指定路径");
}

struct DeviceChoice {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
};

std::string platformString(cl_platform_id platform, cl_platform_info key) {
    size_t size = 0;
    check(clGetPlatformInfo(platform, key, 0, nullptr, &size), "clGetPlatformInfo(size)");
    std::string text(size, '\0');
    check(clGetPlatformInfo(platform, key, size, text.data(), nullptr), "clGetPlatformInfo(value)");
    if (!text.empty() && text.back() == '\0') text.pop_back();
    return text;
}

std::string deviceString(cl_device_id device, cl_device_info key) {
    size_t size = 0;
    check(clGetDeviceInfo(device, key, 0, nullptr, &size), "clGetDeviceInfo(size)");
    std::string text(size, '\0');
    check(clGetDeviceInfo(device, key, size, text.data(), nullptr), "clGetDeviceInfo(value)");
    if (!text.empty() && text.back() == '\0') text.pop_back();
    return text;
}

DeviceChoice chooseDevice() {
    cl_uint platform_count = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &platform_count);
    if (error == kPlatformNotFound || platform_count == 0) {
        throw std::runtime_error("系统没有可用的 OpenCL ICD；WSL 可安装 pocl-opencl-icd 做 CPU 验证");
    }
    check(error, "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> platforms(platform_count);
    check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");

    DeviceChoice fallback;
    for (cl_platform_id platform : platforms) {
        cl_uint count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count) != CL_SUCCESS || count == 0) continue;
        std::vector<cl_device_id> devices(count);
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, count, devices.data(), nullptr), "clGetDeviceIDs");
        for (cl_device_id device : devices) {
            cl_device_type type = 0;
            check(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr), "clGetDeviceInfo(type)");
            if (!fallback.device) fallback = {platform, device};
            if (type & CL_DEVICE_TYPE_GPU) return {platform, device};
        }
    }
    if (!fallback.device) throw std::runtime_error("OpenCL 平台存在，但没有枚举到设备");
    return fallback;
}

struct OpenCLObjects {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_mem weight_q = nullptr;
    cl_mem weight_scale = nullptr;
    cl_mem activation = nullptr;
    cl_mem activation_q = nullptr;
    cl_mem activation_scale = nullptr;
    cl_mem output = nullptr;

    ~OpenCLObjects() {
        if (output) clReleaseMemObject(output);
        if (activation_scale) clReleaseMemObject(activation_scale);
        if (activation_q) clReleaseMemObject(activation_q);
        if (activation) clReleaseMemObject(activation);
        if (weight_scale) clReleaseMemObject(weight_scale);
        if (weight_q) clReleaseMemObject(weight_q);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

template <typename T>
cl_mem makeReadBuffer(cl_context context, const std::vector<T> &values) {
    cl_int error = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   values.size() * sizeof(T), const_cast<T *>(values.data()), &error);
    check(error, "clCreateBuffer(read)");
    return buffer;
}

struct Result {
    std::string name;
    double milliseconds = 0.0;
    float kernel_error = 0.0f;
    float total_error = 0.0f;
};

float maxError(const std::vector<float> &actual, const std::vector<float> &reference) {
    float error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        error = std::max(error, std::fabs(actual[i] - reference[i]));
    }
    return error;
}

Result runKernel(OpenCLObjects &cl, const Options &options, const std::string &name,
                 const std::vector<float> &reference, const std::vector<float> &original) {
    cl_int error = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(cl.program, name.c_str(), &error);
    check(error, ("clCreateKernel(" + name + ")").c_str());

    int argument = 0;
    check(clSetKernelArg(kernel, argument++, sizeof(cl_mem), &cl.weight_q), "clSetKernelArg(weight_q)");
    check(clSetKernelArg(kernel, argument++, sizeof(cl_mem), &cl.weight_scale), "clSetKernelArg(weight_scale)");
    check(clSetKernelArg(kernel, argument++, sizeof(cl_mem), &cl.activation), "clSetKernelArg(activation)");
    check(clSetKernelArg(kernel, argument++, sizeof(cl_mem), &cl.activation_q), "clSetKernelArg(activation_q)");
    check(clSetKernelArg(kernel, argument++, sizeof(cl_mem), &cl.activation_scale), "clSetKernelArg(activation_scale)");
    check(clSetKernelArg(kernel, argument++, sizeof(cl_mem), &cl.output), "clSetKernelArg(output)");
    check(clSetKernelArg(kernel, argument++, sizeof(int), &options.rows), "clSetKernelArg(rows)");
    check(clSetKernelArg(kernel, argument++, sizeof(int), &options.cols), "clSetKernelArg(cols)");
    check(clSetKernelArg(kernel, argument++, sizeof(float), &options.alpha), "clSetKernelArg(alpha)");
    const size_t scratch_bytes = static_cast<size_t>(options.local_size) * sizeof(float);
    check(clSetKernelArg(kernel, argument++, scratch_bytes, nullptr), "clSetKernelArg(scratch)");

    const bool scalar = name.find("scalar") != std::string::npos;
    const size_t global = scalar ? static_cast<size_t>(options.rows)
                                 : static_cast<size_t>(options.rows) * options.local_size;
    const size_t local = static_cast<size_t>(options.local_size);
    const size_t *local_ptr = scalar ? nullptr : &local;

    for (int i = 0; i < options.warmup; ++i) {
        check(clEnqueueNDRangeKernel(cl.queue, kernel, 1, nullptr, &global, local_ptr, 0, nullptr, nullptr),
              "clEnqueueNDRangeKernel(warmup)");
    }
    check(clFinish(cl.queue), "clFinish(warmup)");

    std::vector<cl_event> events(options.iterations);
    for (int i = 0; i < options.iterations; ++i) {
        check(clEnqueueNDRangeKernel(cl.queue, kernel, 1, nullptr, &global, local_ptr, 0, nullptr, &events[i]),
              "clEnqueueNDRangeKernel(profile)");
    }
    check(clFinish(cl.queue), "clFinish(profile)");

    double nanoseconds = 0.0;
    for (cl_event event : events) {
        cl_ulong begin = 0, end = 0;
        check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(begin), &begin, nullptr),
              "clGetEventProfilingInfo(start)");
        check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
              "clGetEventProfilingInfo(end)");
        nanoseconds += static_cast<double>(end - begin);
        clReleaseEvent(event);
    }

    std::vector<float> output(options.rows);
    check(clEnqueueReadBuffer(cl.queue, cl.output, CL_TRUE, 0, output.size() * sizeof(float), output.data(),
                              0, nullptr, nullptr),
          "clEnqueueReadBuffer(output)");
    clReleaseKernel(kernel);
    return {name, nanoseconds / options.iterations / 1.0e6,
            maxError(output, reference), maxError(output, original)};
}

void printUsage(const char *program) {
    std::cout << "用法: " << program << " [--quick] [--rows N] [--cols N] [--local N]"
              << " [--warmup N] [--iterations N] [--kernel PATH]\n";
}

int parsePositive(const char *text, const char *name) {
    const long value = std::stol(text);
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(name) + " 必须是正整数");
    }
    return static_cast<int>(value);
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (++i >= argc) throw std::runtime_error(std::string(name) + " 后面少了参数");
            return argv[i];
        };
        if (argument == "--quick") {
            options.rows = 4096;
            options.warmup = 2;
            options.iterations = 10;
        } else if (argument == "--rows") {
            options.rows = parsePositive(next("--rows"), "rows");
        } else if (argument == "--cols") {
            options.cols = parsePositive(next("--cols"), "cols");
        } else if (argument == "--local") {
            options.local_size = parsePositive(next("--local"), "local");
        } else if (argument == "--warmup") {
            options.warmup = parsePositive(next("--warmup"), "warmup");
        } else if (argument == "--iterations") {
            options.iterations = parsePositive(next("--iterations"), "iterations");
        } else if (argument == "--kernel") {
            options.kernel_path = next("--kernel");
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("不认识的参数: " + argument);
        }
    }
    if (options.cols % kBlockSize != 0) throw std::runtime_error("cols 必须是 32 的倍数");
    if ((options.local_size & (options.local_size - 1)) != 0) {
        throw std::runtime_error("local 必须是 2 的幂，树形归约才不会漏数据");
    }
    return options;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const auto [kernel_path, kernel_source] = findKernel(options, argv[0]);
        const DeviceChoice choice = chooseDevice();

        size_t max_work_group = 0;
        check(clGetDeviceInfo(choice.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group),
                              &max_work_group, nullptr),
              "clGetDeviceInfo(max work-group)");
        if (static_cast<size_t>(options.local_size) > max_work_group) {
            throw std::runtime_error("local 超过设备允许的最大 work-group 大小 " + std::to_string(max_work_group));
        }

        std::cout << "OpenCL 平台: " << platformString(choice.platform, CL_PLATFORM_NAME) << '\n'
                  << "OpenCL 设备: " << deviceString(choice.device, CL_DEVICE_NAME) << '\n'
                  << "问题规模: M=" << options.rows << ", K=" << options.cols
                  << ", local=" << options.local_size << '\n'
                  << "内核文件: " << kernel_path << "\n正在生成并量化测试数据...\n";

        const DataSet data = makeData(options.rows, options.cols, options.alpha);
        OpenCLObjects cl;
        cl_int error = CL_SUCCESS;
        cl.context = clCreateContext(nullptr, 1, &choice.device, nullptr, nullptr, &error);
        check(error, "clCreateContext");
        cl.queue = clCreateCommandQueue(cl.context, choice.device, CL_QUEUE_PROFILING_ENABLE, &error);
        check(error, "clCreateCommandQueue");
        const char *source = kernel_source.c_str();
        const size_t source_size = kernel_source.size();
        cl.program = clCreateProgramWithSource(cl.context, 1, &source, &source_size, &error);
        check(error, "clCreateProgramWithSource");
        error = clBuildProgram(cl.program, 1, &choice.device, "-cl-std=CL1.2", nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(cl.program, choice.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(cl.program, choice.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            throw std::runtime_error("OpenCL 内核编译失败:\n" + log);
        }

        cl.weight_q = makeReadBuffer(cl.context, data.weight_q);
        cl.weight_scale = makeReadBuffer(cl.context, data.weight_scale);
        cl.activation = makeReadBuffer(cl.context, data.activation);
        cl.activation_q = makeReadBuffer(cl.context, data.activation_q);
        cl.activation_scale = makeReadBuffer(cl.context, data.activation_scale);
        cl.output = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY,
                                   static_cast<size_t>(options.rows) * sizeof(float), nullptr, &error);
        check(error, "clCreateBuffer(output)");

        std::vector<Result> results;
        results.push_back(runKernel(cl, options, "gemv_q8_fp16_scalar",
                                    data.reference_weight_q, data.reference_original));
        results.push_back(runKernel(cl, options, "gemv_q8_fp16_local",
                                    data.reference_weight_q, data.reference_original));
        results.push_back(runKernel(cl, options, "gemv_q8_q8_local",
                                    data.reference_both_q, data.reference_original));
        results.push_back(runKernel(cl, options, "gemv_q8_q8_dot4_local",
                                    data.reference_both_q, data.reference_original));

        const double baseline = results.front().milliseconds;
        std::cout << "\n" << std::left << std::setw(25) << "kernel"
                  << std::right << std::setw(12) << "avg ms"
                  << std::setw(12) << "speedup"
                  << std::setw(18) << "kernel max err"
                  << std::setw(18) << "total max err" << '\n';
        for (const Result &result : results) {
            std::cout << std::left << std::setw(25) << result.name
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(12) << result.milliseconds
                      << std::setw(11) << baseline / result.milliseconds << "x"
                      << std::scientific << std::setprecision(3)
                      << std::setw(18) << result.kernel_error
                      << std::setw(18) << result.total_error << '\n';
        }
        std::cout << "\nkernel max err 对比对应量化 CPU 参考；total max err 还包含量化误差。\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "错误: " << exception.what() << '\n';
        return 1;
    }
}
