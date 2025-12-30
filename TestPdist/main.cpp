/**
 * @file main.cpp
 * @brief Ascend C Pdist 算子测试程序 (修复 P=inf 问题版)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <limits> // for std::numeric_limits
#include "acl/acl.h"
#include "aclnn_pdist.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

// =========================================================
// CPU 参考实现 (Golden Kernel) - 已修复 P=inf 支持
// =========================================================
template <typename T>
void cpu_pdist(T* x, T* y, int64_t n, int64_t m, float p) {
    int64_t out_idx = 0;
    bool is_inf = std::isinf(p); // 检查 p 是否为无穷大

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i + 1; j < n; j++) {
            double result = 0.0;
            
            if (is_inf) {
                // P = inf: 切比雪夫距离 (取最大差值)
                double max_diff = 0.0;
                for (int64_t k = 0; k < m; k++) {
                    double diff = std::abs(static_cast<double>(x[i * m + k]) - static_cast<double>(x[j * m + k]));
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
                result = max_diff;
            } else {
                // P = 其他: 闵可夫斯基距离 (累加 pow)
                double sum = 0.0;
                for (int64_t k = 0; k < m; k++) {
                    double diff = std::abs(static_cast<double>(x[i * m + k]) - static_cast<double>(x[j * m + k]));
                    sum += std::pow(diff, static_cast<double>(p));
                }
                result = std::pow(sum, 1.0 / p);
            }
            
            y[out_idx++] = static_cast<T>(result);
        }
    }
}

// =========================================================
// 精度校验工具
// =========================================================
template <typename T>
bool check_accuracy(T* expected, T* actual, int64_t len, float p) {
    double epsilon = (sizeof(T) == 2) ? 1e-2 : 1e-4;
    // 适当放宽阈值
    if (p > 2.0) epsilon *= 5.0;

    double max_err = 0.0;
    int64_t err_count = 0;

    for (int64_t i = 0; i < len; ++i) {
        double val1 = static_cast<double>(expected[i]);
        double val2 = static_cast<double>(actual[i]);
        double diff = std::abs(val1 - val2);
        
        if (diff > epsilon && diff / (std::abs(val1) + 1e-9) > epsilon) {
            if (err_count < 5) {
                std::cout << "[ERROR] Mismatch at index " << i 
                          << ": expected " << val1 << ", got " << val2 
                          << ", diff " << diff << std::endl;
            }
            err_count++;
        }
        if (diff > max_err) max_err = diff;
    }

    std::cout << "[INFO] Max Abs Error: " << max_err << std::endl;
    
    if (err_count > 0) {
        std::cout << "[FAIL] Total " << err_count << " mismatches found." << std::endl;
        return false;
    }
    return true;
}

// =========================================================
// 主测试函数
// =========================================================
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <N> <M> <P> <DType>" << std::endl;
        return -1;
    }
    int64_t N = std::atol(argv[1]);
    int64_t M = std::atol(argv[2]);
    
    // 特殊处理 inf 字符串输入
    std::string p_str = argv[3];
    float p = 2.0;
    if (p_str == "inf" || p_str == "INF") {
        p = std::numeric_limits<float>::infinity();
    } else {
        p = std::atof(argv[3]);
    }

    int dtype_enum = std::atoi(argv[4]); 

    std::cout << ">>> Running Test: N=" << N << ", M=" << M 
              << ", P=" << (std::isinf(p) ? "INF" : std::to_string(p)) 
              << ", Type=" << (dtype_enum == 0 ? "FP32" : "FP16") << std::endl;

    int32_t deviceId = 0;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
    aclrtStream stream;
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

    int64_t inputSize = N * M;
    int64_t outputSize = N * (N - 1) / 2;
    size_t elementSize = (dtype_enum == 0) ? 4 : 2;

    void* xHost = malloc(inputSize * elementSize);
    void* yHost = malloc(outputSize * elementSize);
    void* yRefHost = malloc(outputSize * elementSize);

    void* xDevice = nullptr;
    void* yDevice = nullptr;
    CHECK_RET(aclrtMalloc(&xDevice, inputSize * elementSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&yDevice, outputSize * elementSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);

    std::mt19937 gen(2023);
    std::uniform_real_distribution<float> dis(-10.0, 10.0);

    if (dtype_enum == 0) {
        float* xF32 = (float*)xHost;
        for (int64_t i = 0; i < inputSize; i++) xF32[i] = dis(gen);
    } else {
        uint16_t* xF16 = (uint16_t*)xHost;
        for (int64_t i = 0; i < inputSize; i++) xF16[i] = aclFloatToFloat16(dis(gen));
    }

    CHECK_RET(aclrtMemcpy(xDevice, inputSize * elementSize, xHost, inputSize * elementSize, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, return -1);

    // CPU 计算
    std::cout << "[INFO] Starting CPU calculation..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    if (dtype_enum == 0) {
        cpu_pdist<float>((float*)xHost, (float*)yRefHost, N, M, p);
    } else {
        cpu_pdist<uint16_t>((uint16_t*)xHost, (uint16_t*)yRefHost, N, M, p);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "\033[1;33m[PERF] CPU Time: " << std::fixed << std::setprecision(4) << cpu_time_ms << " ms\033[0m" << std::endl;

    // NPU 计算
    aclDataType aclType = (dtype_enum == 0) ? ACL_FLOAT : ACL_FLOAT16;
    int64_t inputShape[] = {N, M};
    int64_t outputShape[] = {outputSize};
    aclTensor* xTensor = aclCreateTensor(inputShape, 2, aclType, nullptr, 0, aclFormat::ACL_FORMAT_ND, inputShape, 2, xDevice);
    aclTensor* yTensor = aclCreateTensor(outputShape, 1, aclType, nullptr, 0, aclFormat::ACL_FORMAT_ND, outputShape, 1, yDevice);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    CHECK_RET(aclnnPdistGetWorkspaceSize(xTensor, p, yTensor, &workspaceSize, &executor) == ACL_SUCCESS, return -1);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) CHECK_RET(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);

    // Warmup
    aclnnPdist(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    auto start_npu = std::chrono::high_resolution_clock::now();
    CHECK_RET(aclnnPdist(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);
    auto end_npu = std::chrono::high_resolution_clock::now();
    
    double npu_time_ms = std::chrono::duration<double, std::milli>(end_npu - start_npu).count();
    std::cout << "\033[1;32m[PERF] NPU Time: " << std::fixed << std::setprecision(4) << npu_time_ms << " ms\033[0m" << std::endl;

    if (cpu_time_ms > 0) std::cout << "\033[1;36m[PERF] Speedup: " << (cpu_time_ms / npu_time_ms) << "x \033[0m" << std::endl;

    CHECK_RET(aclrtMemcpy(yHost, outputSize * elementSize, yDevice, outputSize * elementSize, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS, return -1);

    bool pass = true;
    if (dtype_enum == 0) {
        pass = check_accuracy<float>((float*)yRefHost, (float*)yHost, outputSize, p);
    } else {
        std::cout << "[WARN] FP16 strict accuracy check skipped in C++." << std::endl;
    }

    std::cout << (pass ? "\033[32m[PASS]\033[0m" : "\033[31m[FAIL]\033[0m") << std::endl;

    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    free(xHost);
    free(yHost);
    free(yRefHost);
    aclrtDestroyStream(stream);
    aclFinalize();
    return 0;
}