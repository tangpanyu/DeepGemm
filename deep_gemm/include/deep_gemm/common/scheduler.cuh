#pragma once

#include <deep_gemm/common/types.hpp>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

/**
 * @brief 索引类型枚举
 * 用于指定 get_global_idx 函数计算哪个维度的全局索引
 */
enum class IndexType {
    MN,    // M 或 N 维度索引
    K,     // K 维度索引
    SF_K,  // Scale Factor K 维度索引 (用于 FP8 GEMM)
};

/**
 * @brief 编译期计算最优的 1D blocks 分组数量
 * 
 * **目的**: 优化 TMA Multicast 的分组策略，最小化 Shared Memory 使用
 * 
 * **候选值**: 8 或 16 个 blocks 为一组
 * 
 * **优化目标**: 
 * - kIsMulticastOnA=true:  最小化 (candidate * BLOCK_N + ceil(kNumSMs/candidate) * BLOCK_M)
 * - kIsMulticastOnA=false: 最小化 (candidate * BLOCK_M + ceil(kNumSMs/candidate) * BLOCK_N)
 * 
 * @tparam kGemmType - GEMM 类型
 * @tparam BLOCK_M, N - Block 维度
 * @tparam kNumSMs - GPU 的 SM 数量
 * @tparam kIsMulticastOnA - Multicast 是否应用于 A 矩阵
 * @return uint32_t - 最优的 blocks 每组数量 (8 或 16)
 */
template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // 从候选值中选择最优
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto& candidate: {8u, 16u}) {
        const auto& usage = kIsMulticastOnA ?
                    candidate * BLOCK_N + constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // N 维度分组
                    candidate * BLOCK_M + constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // M 维度分组
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"

/**
 * @brief Block 调度器 - 负责将 GEMM blocks 分配到 GPU SMs
 * 
 * **核心功能**:
 * 1. 持久化调度 (Persistent Scheduling): 每个 SM 持续处理多个 blocks
 * 2. Swizzling: 重排 block 顺序以优化 L2 cache 利用率
 * 3. TMA Multicast 支持: 多个 CTAs 共享一次 TMA 加载
 * 4. 多 GEMM 变体支持: Normal, M-Grouped, K-Grouped, Batched 等
 * 
 * **调度策略**:
 * - 不是 Hilbert 曲线或 Morton 曲线
 * - 而是 "分组 swizzled 行/列优先" (Grouped Swizzled Row/Column-Major)
 * - 简单高效，适合 GEMM 的规则访问模式
 * 
 * @tparam kGemmType - GEMM 类型
 * @tparam BLOCK_M, N - Block 维度
 * @tparam kNumGroups - MoE 分组数量
 * @tparam kNumMulticast - TMA Multicast 数量 (1 或 2)
 * @tparam kIsMulticastOnA - Multicast 应用于 A (true) 或 B (false)
 * @tparam kNumSMs - 目标 GPU 的 SM 数量
 * @tparam SF_K_ALIGNMENT - Scale Factor K 维度对齐 (SM90=128, SM100=512)
 * @tparam kNum1DBlocksPerGroup - 每组 1D blocks 数量 (编译期计算最优值)
 */
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t SF_K_ALIGNMENT = 512u,  // 仅用于 k-grouped GEMM: 128 (SM90 float SF) 或 512 (SM100 UE8M0 SF)
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {
    // ========== 状态变量 ==========
    
    /**
     * @brief 当前调度迭代次数
     * 每次迭代处理 kNumSMs 个 blocks (每个 SM 一个)
     */
    int current_iter = -1;

    // ========== Block 配置 ==========
    /**
     * @brief Total blocks 数量
     * - Normal/Batched: num_m_blocks * num_n_blocks
     * - Grouped: 动态计算
     */
    uint32_t num_blocks;
    
    /**
     * @brief M 维度的 blocks 数量
     * = ceil(shape_m / BLOCK_M)
     */
    uint32_t num_m_blocks;
    
    /**
     * @brief N 维度的 blocks 数量
     * = ceil(shape_n / BLOCK_N)
     */
    uint32_t num_n_blocks;

    // ========== SM90 Multicast 检查 ==========
    /**
     * @brief 当前组中的 blocks 数量
     * 用于 TMA Multicast 有效性检查
     */
    uint32_t num_blocks_in_group;
    
    /**
     * @brief 相邻 CTA 是否存活
     * 用于决定 Multicast 目标
     * - true: 可以发送到相邻 CTA
     * - false: 只能发送到当前 CTA
     */
    bool is_peer_cta_alive = true;

    // ========== Grouped GEMM 状态 ==========
    /**
     * @brief Grouped layout 数组指针
     * 存储每个 group 的 token 数量或 offset
     */
    int* grouped_layout;
    
    /**
     * @brief 当前处理的 group 索引
     * 用于 M-Grouped 和 K-Grouped GEMMs
     */
    uint32_t current_group_idx = 0;
    
    /**
     * @brief M 维度 block 累积和 (仅用于 Masked Layout)
     * 用于追踪已处理的 M blocks
     */
    uint32_t current_m_cumsum = 0;
    
    /**
     * @brief Psum Layout 相关状态 (仅用于 Contiguous Psum Layout)
     * - last_psum_m: 上一个 group 的 psum_m (对齐到 128)
     * - current_psum_m: 当前 group 的 psum_m
     * - current_m_block_cumsum: M block 累积和
     */
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    
    /**
     * @brief K-Grouped Layout 相关状态
     * - current_shape_k: 当前 group 的 K 维度
     * - current_num_valid_groups: 有效 groups 数量
     * - current_k_cumsum: K 维度累积和
     * - current_sf_k_cumsum: Scale Factor K 维度累积和
     * - next_group_idx: 下一个 group 索引
     * - next_shape_k: 下一个 group 的 K 维度
     */
    uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, current_sf_k_cumsum = 0;
    uint32_t next_group_idx, next_shape_k;

    // ========== K-Grouped GEMM 辅助函数 ==========
    /**
     * @brief 查找下一个有效的 K group
     * 跳过 shape_k = 0 的空 groups
     * 
     * @param group_idx - [输入/输出] 当前 group 索引，更新为下一个有效索引
     * @param shape_k - [输出] 找到的 shape_k 值
     */
    __device__ __forceinline__ void get_next_k_group(uint32_t &group_idx, uint32_t &shape_k) const {
        for (; group_idx < kNumGroups; ++ group_idx) {
            shape_k = __ldg(grouped_layout + group_idx);  // 从 grouped_layout 读取
            if (shape_k > 0)  // 找到非空 group
                break;
        }
    }

    // ========== 构造函数 ==========
    /**
     * @brief Scheduler 构造函数
     * 初始化所有状态变量
     * 
     * @param shape_m, shape_n, shape_k - 矩阵维度
     * @param grouped_layout - Grouped layout 数组 (可选，用于 Grouped GEMMs)
     */
    // ReSharper disable once CppPossiblyUninitializedMember
    __device__ __forceinline__ explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n, const uint32_t& shape_k,
                                                  int* grouped_layout = nullptr) {
        num_m_blocks = ceil_div(shape_m, BLOCK_M);
        num_n_blocks = ceil_div(shape_n, BLOCK_N);
        current_shape_k = shape_k;
        
        // 根据 GEMM 类型初始化
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            this->grouped_layout = grouped_layout;
            current_psum_m = __ldg(grouped_layout);  // 读取第一个 group 的 psum_m
            num_m_blocks = ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            this->grouped_layout = grouped_layout;
            get_next_k_group(current_group_idx, current_shape_k);  // 查找第一个有效 group
            next_group_idx = current_group_idx + 1;
            get_next_k_group(next_group_idx, next_shape_k);  // 预取下一个 group
        }
    }

    // ========== Swizzled Block 索引计算 ==========
    /**
     * @brief 计算 swizzled block 索引 (M/N 坐标)
     * 
     * **Swizzling 目的**: 优化 L2 cache 利用率
     * 
     * **算法流程**:
     * 1. 计算 group 索引和组内偏移
     * 2. 处理未对齐的 TMA Multicast (仅 SM90)
     * 3. 转换为最终的 M/N block 索引
     * 
     * **分组策略**:
     * - kIsMulticastOnA=true:  N 维度分组 (groups on N)
     * - kIsMulticastOnA=false: M 维度分组 (groups on M)
     * 
     * @param block_idx - 线性 block 索引
     * @param m_block_idx - [输出] M 维度 block 索引
     * @param n_block_idx - [输出] N 维度 block 索引
     */
    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle 优化 L2 利用率
        const auto& primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;    // 主要维度
        const auto& secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;  // 次要维度
        const auto& num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;    // 每组 blocks 总数
        const auto& group_idx = block_idx / num_blocks_per_group;                          // 组索引
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;                           // 组起始索引
        auto in_group_idx = block_idx % num_blocks_per_group;                              // 组内偏移
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);  // 实际组大小

        // ========== 修复未对齐的 TMA Multicast ==========
        /**
         * @brief 处理奇数行的 TMA Multicast
         * 
         * **仅用于 SM90**:
         * - SM90: 可以动态禁用 TMA Multicast
         * - SM100: 使用固定 2-CTA，无法动态禁用
         * 
         * **修复逻辑**:
         * 如果 num_blocks_in_group 是奇数，调整为偶数
         * - 前半部分：使用 num_blocks_in_group ^ 1 (减 1)
         * - 后半部分：剩余的 1 个 block 单独处理
         */
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;  // 调整为偶数
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;  // 剩余 1 个 block
            }
        }
#endif

        // ========== 转换为最终 M/N block 索引 ==========
        /**
         * @brief 根据 Multicast 配置计算最终坐标
         * 
         * **映射规则**:
         * - kIsMulticastOnA=true:  N 维度分组 → m_block_idx = 组内索引/组大小, n_block_idx = 组起始 + 组内偏移
         * - kIsMulticastOnA=false: M 维度分组 → m_block_idx = 组起始 + 组内偏移, n_block_idx = 组内索引/组大小
         */
        // `kIsMulticastOnA == true` 导致 N 维度分组
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    // ========== 全局索引计算 ==========
    /**
     * @brief 计算全局索引 (考虑 group offset)
     * 
     * **模板参数**:
     * @tparam kWithGroupOffset - 是否加上 group offset
     * @tparam kIndexType - 索引类型 (MN, K, SF_K)
     * 
     * **不同 GEMM 类型的 offset 计算**:
     * - Normal: offset = 0
     * - MGroupedContiguous: offset = grouped_layout[m_block_idx * BLOCK_M]
     * - MGroupedMasked/Psum: offset = current_group_idx
     * - KGroupedContiguous: offset = current_group_idx * shape_dim (MN) 或 current_k_cumsum (K)
     * - Batched: offset = current_group_idx (仅 SF_K)
     * 
     * @param shape_dim - 维度大小
     * @param block_size - Block 大小
     * @param block_idx - Block 索引
     * @param m_block_idx - M block 索引 (用于 MGroupedContiguous)
     * @return uint32_t - 全局索引
     */
    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            const auto offset = kWithGroupOffset ? cute::max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M)) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            auto offset = 0;
            if constexpr (kWithGroupOffset) {
                if constexpr (kIndexType == IndexType::MN)
                    offset = current_group_idx * shape_dim;
                else if constexpr (kIndexType == IndexType::K)
                    offset = current_k_cumsum;
                else if constexpr (kIndexType == IndexType::SF_K)
                    offset = current_sf_k_cumsum;
            }
            return offset + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::Batched) {
            // 忽略 kWithGroupOffset，仅对 SF_K 应用 offset
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // ========== Block 调度主函数 ==========
    /**
     * @brief 获取下一个 block 的 M/N 坐标
     * 
     * **持久化调度算法**:
     * 1. 计算下一个全局 block 索引：next_block_idx = current_iter * kNumSMs + blockIdx.x
     * 2. 根据 GEMM 类型处理不同的调度逻辑
     * 3. 返回是否还有 blocks 需要处理
     * 
     * **支持的 GEMM 类型**:
     * - MGroupedMasked: Masked M-Grouped GEMM (MoE forward)
     * - MGroupedContiguousWithPsumLayout: Psum Layout 优化的 M-Grouped
     * - KGroupedContiguous: K-Grouped GEMM (MoE backward)
     * - Batched: Batched GEMM
     * - Normal: 标准 GEMM
     * 
     * @param m_block_idx - [输出] M 维度 block 索引
     * @param n_block_idx - [输出] N 维度 block 索引
     * @return bool - 是否还有有效的 blocks
     */
    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

        // ========== MGroupedMasked 调度 ==========
        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                // 任务结束
                if (current_group_idx == kNumGroups)
                    return false;

                // 当前 group 内的 blocks
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + current_group_idx)), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // 移动到下一个 group
                current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        }

        // ========== MGroupedContiguousWithPsumLayout 调度 ==========
        else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
            while (true) {
                // 当前 group 内的 blocks
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;

                // 移动到下一个 group
                if (++ current_group_idx == kNumGroups)
                    return false;

                // 注意：num_m_blocks 随 group 索引变化
                last_psum_m = align(current_psum_m, 128u);  // 对齐到 128
                current_psum_m = __ldg(grouped_layout + current_group_idx);
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = ceil_div(current_psum_m - last_psum_m, BLOCK_M);
            }

            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);

            // 注意：last_psum_m 已对齐到 128，需要加上偏移
            m_block_idx += last_psum_m / BLOCK_M;
            DG_STATIC_ASSERT(128 % BLOCK_M == 0, "Invalid BLOCK_M");
        }

        // ========== KGroupedContiguous 调度 ==========
        else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            while (true) {
                // 任务结束
                if (current_group_idx == kNumGroups)
                    return false;

                // 当前 group 内的 blocks
                if (next_block_idx < (current_num_valid_groups + 1) * num_m_blocks * num_n_blocks)
                    break;

                // 移动到下一个 group
                current_k_cumsum += current_shape_k;
                current_sf_k_cumsum += ceil_div(current_shape_k, SF_K_ALIGNMENT);
                current_num_valid_groups ++;

                current_group_idx = next_group_idx ++;
                current_shape_k = next_shape_k;
                get_next_k_group(next_group_idx, next_shape_k);  // 预取下一个 group
            }

            get_swizzled_block_idx(next_block_idx - current_num_valid_groups * num_m_blocks * num_n_blocks, m_block_idx, n_block_idx);
        }

        // ========== Batched GEMM 调度 ==========
        else if constexpr (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            current_group_idx = next_block_idx / num_blocks;  // 计算 batch 索引
            const auto& block_idx = next_block_idx - current_group_idx * num_blocks;  // batch 内的 block 索引
            if constexpr (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        }

        // ========== Normal GEMM 调度 ==========
        else {
            if (next_block_idx >= num_blocks)
                return false;

            // 仅用于 SM90
            // 注意：Masked Grouped GEMM 不需要设置 is_peer_cta_alive，因为必须对齐
            is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or                  // N 维度总是对齐 (常量绕过)
                                num_m_blocks % kNumMulticast == 0 or                  // M 维度总是对齐 (常量绕过)
                                (next_block_idx ^ 1) < num_blocks;                    // 相邻 CTA 在边界内
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // ========== TMA Multicast 有效性检查 ==========
    /**
     * @brief 检查 TMA Multicast 是否有效
     * 
     * **无效条件**:
     * 1. num_blocks_in_group == 1: 只有一个 block，无法 multicast
     * 2. MGroupedContiguous 且 kIsMulticastOnA=false: 需要检查相邻 blocks 是否同 group
     * 
     * **有效条件**:
     * - Normal, MGroupedMasked, KGroupedContiguous, Batched: 总是有效
     * - MGroupedContiguous + kIsMulticastOnA=true: 总是有效
     * - MGroupedContiguous + kIsMulticastOnA=false: 需要相邻 blocks 同 group
     * 
     * @param m_block_idx - M block 索引
     * @return bool - TMA Multicast 是否有效
     */
    // 仅用于 SM90
    __device__ __forceinline__ bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        
        // 这些类型总是支持 Multicast
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;  // Multicast on A 总是有效
            } else {
                // Multicast on B 需要检查相邻 blocks 是否同 group
                const auto& group_idx = __ldg(grouped_layout + m_block_idx * BLOCK_M);
                const auto& peer_group_idx = __ldg(grouped_layout + (m_block_idx ^ 1) * BLOCK_M);
                return group_idx == peer_group_idx;  // 相同 group 才有效
            }
        }
    }

    // ========== 计算有效性检查 ==========
    /**
     * @brief 检查当前 block 的计算是否有效
     * 
     * **不同 GEMM 类型的检查**:
     * - Normal/Batched: 总是有效
     * - MGroupedContiguous: 检查 grouped_layout 值 >= 0
     * - MGroupedMasked: 检查 m_offset + m_block_idx * BLOCK_M < token 数量
     * 
     * @param m_block_idx - M block 索引
     * @param m_offset - M 维度 offset
     * @return bool - 计算是否有效
     */
    // 仅用于 SM90
    // ReSharper disable once CppNotAllPathsReturnValue
    __device__ __forceinline__ bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;  // 标准 GEMM 总是有效
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return __ldg(grouped_layout + m_offset + m_block_idx * BLOCK_M) >= 0;  // 检查 token 数量
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < __ldg(grouped_layout + current_group_idx);  // 检查边界
        } else {
            // 不可达代码
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm
