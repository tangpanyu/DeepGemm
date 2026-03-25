#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cute/container/tuple.hpp>

#include "cute_tie.cuh"

#ifdef __CLION_IDE__

/**
 * @brief IDE 环境下的 printf 替代实现
 * CLion 无法正确识别 CUDA 的 printf，用 trap 指令避免编译错误
 */
__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

/**
 * @brief 设备端断言宏 (完整版)
 * 失败时打印错误信息并触发 trap 指令
 */
#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

/**
 * @brief 设备端断言宏 (精简版)
 * 仅触发 trap，不打印信息 (用于性能敏感场景)
 */
#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

/**
 * @brief 编译期断言宏
 * 封装 static_assert，支持自定义错误消息
 */
#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

namespace deep_gemm {

/**
 * @brief 模式访问函子包装器
 * 将 lambda 或函数对象包装成可通过 operator[] 访问的模式
 * 
 * @tparam FuncT - 函子类型
 * @param func - 接受索引 i 并返回对应值的函数
 */
template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    __device__ __host__
    auto operator [](const uint32_t& i) {
        return func(i);
    }
};

/**
 * @brief 向上取整除法 (运行时版本)
 * 计算 ceil(a / b)，等价于 (a + b - 1) / b
 * 
 * @tparam T - 整数类型
 */
template <typename T>
__device__ __host__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

/**
 * @brief 向上取整除法 (编译期版本)
 * constexpr 版本，可在编译期求值
 * 
 * @tparam T - 整数类型
 */
template <typename T>
__device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

/**
 * @brief 对齐到 b 的倍数 (运行时版本)
 * 计算大于等于 a 的最小 b 的倍数
 * 
 * @tparam T - 整数类型
 */
template <typename T>
__device__ __host__ T align(T a, T b) {
    return ceil_div(a, b) * b;
}

/**
 * @brief 对齐到 b 的倍数 (编译期版本)
 * constexpr 版本，可在编译期求值
 * 
 * @tparam T - 整数类型
 */
template <typename T>
__device__ __host__ constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

/**
 * @brief 计算最大公约数 (编译期递归版本)
 * 使用欧几里得算法，constexpr 可在编译期求值
 * 
 * @tparam T - 整数类型
 */
template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

/**
 * @brief 交换两个变量的值
 * 内联设备函数，使用临时变量交换
 * 
 * @tparam T - 变量类型
 */
template<typename T>
__forceinline__ __device__ void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

/**
 * @brief 获取当前 SM (Streaming Multiprocessor) 索引
 * 通过读取 %%smid 寄存器获取当前 thread block 所在的 SM ID
 * 
 * @return uint32_t - SM 索引号
 */
__forceinline__ __device__ uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

/**
 * @brief 获取当前 lane 在 warp 中的索引
 * 通过读取 %%laneid 寄存器获取线程 ID
 * 
 * @return uint32_t - Lane ID (0-31)
 */
__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

/**
 * @brief 从 shared memory 加载 uint32_t (内联汇编版本)
 * 使用 PTX ld.shared 指令，比常规 load 更高效
 * 
 * @param ptr - shared memory 指针
 * @return uint32_t - 加载的值
 */
__device__  __forceinline__ uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

/**
 * @brief 从 shared memory 加载 float2 (向量加载)
 * 使用 v2.f32 指令一次加载两个 float
 * 
 * @param ptr - shared memory 指针
 * @return float2 - 加载的两个浮点数
 */
__device__  __forceinline__ float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

/**
 * @brief 从 shared memory 加载 float4 (向量加载)
 * 使用 v4.f32 指令一次加载四个 float
 * 
 * @param ptr - shared memory 指针
 * @return float4 - 加载的四个浮点数
 */
__device__  __forceinline__ float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

/**
 * @brief 从 shared memory 加载 uint4 (向量加载)
 * 使用 v4.u32 指令一次加载四个 uint32_t
 * 
 * @param ptr - shared memory 指针
 * @return uint4 - 加载的四个无符号整数
 */
__device__  __forceinline__ uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

/**
 * @brief 从 shared memory 加载单个 float
 * 使用 ld.shared.f32 指令
 * 
 * @param ptr - shared memory 指针
 * @return float - 加载的浮点值
 */
__device__  __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

/**
 * @brief 存储 float 到 shared memory
 * 使用 st.shared.f32 指令
 * 
 * @param ptr - shared memory 指针
 * @param val - 要存储的浮点值
 */
__device__ __forceinline__ void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val));
}

/**
 * @brief 存储 float2 到 shared memory (向量存储)
 * 使用 st.shared.v2.f32 指令一次存储两个 float
 * 
 * @param ptr - shared memory 指针
 * @param val - 要存储的 float2 值
 */
__device__ __forceinline__ void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

/**
 * @brief 存储 uint32_t 到 shared memory
 * 使用 st.shared.u32 指令
 * 
 * @param ptr - shared memory 指针
 * @param val - 要存储的无符号整数
 */
__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

/**
 * @brief 存储两个 uint32_t 到 shared memory (向量存储)
 * 使用 st.shared.v2.u32 指令一次存储两个 uint32
 * 
 * @param ptr - shared memory 指针
 * @param x - 第一个值
 * @param y - 第二个值
 */
__device__  __forceinline__ void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y));
}

/**
 * @brief 存储四个 uint32_t 到 shared memory (向量存储)
 * 使用 st.shared.v4.u32 指令一次存储四个 uint32
 * 
 * @param ptr - shared memory 指针
 * @param x - 第一个值
 * @param y - 第二个值
 * @param z - 第三个值
 * @param w - 第四个值
 */
__device__  __forceinline__ void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

/**
 * @brief 存储 __int128_t 到 shared memory
 * 使用 st.shared.b128 指令存储 128 位整数
 * 
 * @param ptr - shared memory 指针
 * @param val - 要存储的 128 位整数
 */
__device__ __forceinline__ void st_shared(const __int128_t* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}

/**
 * @brief 将两个值转换为 BF16 并打包成 int32
 * 将两个 old_t 类型的值转换为 float，再转为 BF16，最后打包成 32 位整数
 * 
 * @tparam old_t - 原始类型 (通常是 float)
 * @param x - 第一个值 (引用)
 * @param y - 第二个值 (引用)
 * @return int - 打包后的 32 位整数 (两个 BF16)
 */
template <typename old_t>
__device__ __forceinline__ int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
    return *reinterpret_cast<int*>(&bf16x2);
}

/**
 * @brief 预取数据到 L1 cache
 * 使用 prefetch.global.L1 指令将全局内存数据预取到 L1 缓存
 * 
 * @param ptr - 要预取的内存指针
 */
__device__ __forceinline__ void prefetch_l1(void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

/**
 * @brief 向量化加载辅助结构体
 * 根据字节数自动选择合适的向量类型 (uint4, uint2, uint32_t)
 * 
 * @tparam kNumBytes - 要加载的字节数
 * 
 * 示例:
 * - 16 bytes → uint4 (ulonglong4)
 * - 8 bytes  → uint2
 * - 4 bytes  → uint32_t
 */
template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

} // namespace `deep_gemm`
