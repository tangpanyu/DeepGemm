#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

/**
 * @file sm90_bf16_gemm.cuh
 * @brief SM90 (Hopper 架构) BF16 GEMM 核心实现
 * 
 * **支持特性**:
 * - BF16 (Brain Floating Point 16-bit) 精度
 * - TMA (Tensor Memory Accelerator) 异步数据传输
 * - WGMMA (Warp Group Matrix Multiply-Accumulate) 指令
 * - Software Pipelining (多级流水线)
 * - TMA Multicast (多个 CTAs 共享数据)
 * - 支持多种 GEMM 变体 (Normal, M-Grouped, K-Grouped, Batched)
 * 
 * **与 FP8 GEMM 的区别**:
 * - 数据类型：BF16 (2 bytes) vs FP8 (1 byte)
 * - 精度更高，但内存带宽需求更大
 * - 不需要处理 scaling factor
 */

/**
 * @brief SM90 BF16 GEMM 核心实现 kernel
 * 
 * **模板参数**:
 * @tparam kMajorA - A 矩阵的内存布局 (K 或 MN)
 * @tparam kMajorB - B 矩阵的内存布局 (K 或 MN)
 * @tparam SHAPE_M, N, K - 全局矩阵维度 (编译期常量，0 表示运行时动态)
 * @tparam kNumGroups - MoE 分组数量 (Normal GEMM 时为 1)
 * @tparam BLOCK_M, N, K - Thread block 级别的分块大小
 * @tparam kSwizzleAMode, BMode, DMode - Shared memory 交织模式 (0=无，>0=XOR 交织)
 * @tparam kNumStages_ - 主流水线的级数 (通常 4-10 级)
 * @tparam kNumTMAThreads - TMA warp 的线程数 (通常 128)
 * @tparam kNumMathThreads - Math warp 的线程数 (128 或 256)
 * @tparam kNumTMAMulticast - TMA Multicast 数量 (1 或 2)
 * @tparam kIsTMAMulticastOnA - Multicast 是否应用于 A 矩阵 (true=A, false=B)
 * @tparam kNumSMs - 目标 GPU 的 SM 数量
 * @tparam kGemmType - GEMM 类型 (Normal, MGrouped, KGrouped, Batched 等)
 * @tparam kWithAccumulation - TMA store 是否使用累加模式 (reduce-add)
 * @tparam cd_dtype_t - 输出数据类型 (BF16 或 float)
 * 
 * **输入参数**:
 * @param grouped_layout - MoE 分组布局数组 (每个 expert 的 token 数或 offset)
 * @param shape_m, n, k - 实际矩阵维度 (运行时值)
 * @param tensor_map_a, b, cd - TMA 描述符 (grid_constant，所有 SMs 共享)
 */
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages_,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bf16_gemm_impl(int* grouped_layout,
                    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    /**
     * @brief Stage 合并优化 (仅 Normal GEMM + NT 布局)
     * 
     * **目的**: 减少 warpgroup_wait<0>() 的同步开销
     * **方法**: 将多个小 stage 合并成一个大 stage
     * 
     * 示例:
     * - 原始：kNumStages_ = 10, BLOCK_K_ = 64
     * - 合并后：kNumStages = 5, BLOCK_K = 128 (每个 stage 处理 2 个原始 stage)
     */
    constexpr uint32_t kDoMergeStages =
        kNumStages_ >= 10 and
        kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K and
        kNumMathThreads == 128;
    // 确保合并后至少有 kNumMinStages 个 stages
    constexpr uint32_t kNumMinStages = 5;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;  // 合并后的实际 BLOCK_K
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;  // 合并后的 stage 数

    // ========== 类型定义 ==========
    /**
     * @brief WGMMA 类型选择器
     * 根据 BLOCK_N 和布局自动选择合适的 WGMMA 指令变体
     */
    using WGMMA = typename BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // ========== 覆盖编译期常量 ==========
    /**
     * @brief 如果编译期常量不为 0，则使用编译期值；否则使用运行时参数
     * 这使得同一 kernel 可以支持编译期已知和运行时动态的维度
     */
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // ========== Shared Memory 布局 ==========
    /**
     * @brief Shared memory 大小计算
     * 
     * **布局结构** (总大小 = D + A + B + Barriers):
     * 
     * | 区域 | 大小 | 用途 |
     * |------|------|------|
     * | D | BLOCK_M × BLOCK_N × sizeof(cd_dtype_t) | 输出缓冲区 (BF16 或 float) |
     * | A (per stage) | BLOCK_M × BLOCK_K × 2 | A 矩阵数据 (kNumStages 级) |
     * | B (per stage) | BLOCK_N × BLOCK_K × 2 | B 矩阵数据 (kNumStages 级) |
     * | Barriers | 2 × kNumStages × sizeof(Barrier) | 同步屏障 |
     */
    static constexpr uint32_t SMEM_D_SIZE = constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cd_dtype_t)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    // 确保 WGMMA 有足够的 padding 空间
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // ========== 线程索引和配置 ==========
    /**
     * @brief 获取 warp 和 lane 索引
     * warp_idx: 当前线程所属的 warp ID
     * lane_idx: 当前线程在 warp 中的 ID (0-31)
     */
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // ========== TMA 描述符预取 ==========
    /**
     * @brief 预取 TMA 描述符到 cache
     * 由负责 TMA 的 warp 执行，减少后续访问延迟
     */
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // ========== Shared Memory 分配和对齐 ==========
    /**
     * @brief 分配 shared memory，对齐到 1024 字节
     * 1024 字节对齐是为了支持 swizzle-128B 模式，避免 bank conflict
     */
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B/D must be aligned to 1024 bytes");

    // ========== Shared Memory 指针初始化 ==========
    /**
     * @brief 设置 D/A/B 的 shared memory 指针
     * 
     * **内存布局**:
     * smem_buffer:
     *   [0] -> D 输出缓冲区
     *   [SMEM_D_SIZE] -> A 矩阵 stage 0
     *   [SMEM_D_SIZE + SMEM_A_SIZE_PER_STAGE] -> A 矩阵 stage 1
     *   ...
     *   [SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE] -> B 矩阵 stage 0
     *   [SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE] -> B 矩阵 stage 1
     *   ...
     *   [末尾] -> Barriers
     */
    auto smem_d = reinterpret_cast<cd_dtype_t*>(smem_buffer);
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // ========== Barrier 初始化 ==========
    /**
     * @brief 设置 barrier 指针和模式访问器
     * 
     * **Barrier 布局**:
     * - full_barriers[i]: 第 i 级流水线的"数据就绪"屏障
     *   - TMA 生产者在数据加载完成后 arrive
     *   - Math 消费者等待数据就绪后 wait
     * 
     * - empty_barriers[i]: 第 i 级流水线的"缓冲区空"屏障
     *   - Math 消费者在使用完缓冲区后 arrive
     *   - TMA 生产者等待缓冲区空后 wait
     * 
     * **Multicast 支持**: 每个 empty barrier 需要等待 kNumTMAMulticast 个 CTAs
     */
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers  = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    // 初始化所有 barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);  // 1 个 TMA 生产者
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);  // 多个 Math 消费者
        }

        // 使 barrier 在 async proxy 中可见
        cutlass::arch::fence_barrier_init();
    }

    // 同步所有线程，使 barrier 在普通内存模型中可见
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // ========== 寄存器重新配置 ==========
    /**
     * @brief 为 TMA 和 Math warp 分配不同的寄存器数量
     * 
     * **目的**: 优化寄存器使用，减少 spill
     * - TMA warp: 需要较少寄存器 (48 个)，主要用于 TMA 操作
     * - Math warp: 需要更多寄存器 (224-248 个)，用于 WGMMA 累加器
     */
    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    // ========== Block 调度器初始化 ==========
    /**
     * @brief 初始化 scheduler，负责分配 blocks 到 SMs
     * scheduler 会根据 GEMM 类型 (Normal, M-Grouped, K-Grouped) 计算每个 block 的 M/N 坐标
     */
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // ========== 流水线状态管理 ==========
    /**
     * @brief 流水线阶段和相位管理
     * 
     * **stage_idx**: 当前使用的流水线级 (0 到 kNumStages-1)
     * **phase**: 当前流水线的相位 (0 或 1)，用于双缓冲
     * 
     * **advance_pipeline 逻辑**:
     * 1. k_block_idx++ (进入下一个 K 块)
     * 2. stage_idx 循环递增 (0→1→2→...→kNumStages-1→0)
     * 3. 当 stage_idx 回到 0 时，翻转 phase (0↔1)
     * 
     * 示例 (kNumStages=4):
     * iter: 0  1  2  3  4  5  6  7
     * stage: 0  1  2  3  0  1  2  3
     * phase: 0  0  0  0  1  1  1  1  (翻转)
     */
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // 仅在进入下一个第一轮 stage 时翻转相位
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ========== TMA Warp-Group (Producer) ==========
    /**
     * @brief TMA warp-group 负责从 Global Memory 加载数据到 Shared Memory
     * 
     * **执行路径**:
     * 1. 分配较少的寄存器 (48 个)，因为 TMA 操作简单
     * 2. 使用第 3 个 warp (warp_idx = kNumMathThreads/32 + 2)
     *    - Warp 0-1: 可能在进行 BLOCK_M=32 的 WGMMA
     *    - Warp 2: 专门负责 TMA 加载
     * 3. 持续调度 blocks，直到所有任务完成
     */
    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // 仅一个线程 (或 warp) 执行 TMA
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for TMA warp-group");

            // ========== 持久化调度 ==========
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // ========== TMA Multicast 配置 ==========
                /**
                 * @brief 配置 A 和 B 的 TMA Multicast 数量
                 * 
                 * **Multicast 条件**:
                 * 1. kNumTMAMulticast > 1 (编译期启用)
                 * 2. scheduler.is_tma_multicast_valid() 检查是否有效
                 *    - 检查相邻 blocks 是否属于同一个 MoE group
                 *    - 避免跨 group 的无效 multicast
                 * 
                 * **分配策略**:
                 * - kIsTMAMulticastOnA=true: Multicast 应用于 A 矩阵
                 * - kIsTMAMulticastOnA=false: Multicast 应用于 B 矩阵
                 */
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                // ========== K 维度循环 ==========
                const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // ========== 等待消费者释放缓冲区 ==========
                    /**
                     * @brief 等待 Math warp 完成上一轮相同 stage 的计算
                     * phase ^ 1: 等待上一相位 (生产者落后消费者一个相位)
                     */
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // ========== 计算全局索引 ==========
                    /**
                     * @brief 计算 A/B 矩阵在 Global Memory 中的索引
                     * 
                     * **kWithGroupOffset**:
                     * - MGroupedMasked: 需要加上 group offset
                     * - 其他类型: offset = 0
                     * 
                     * **IndexType**:
                     * - MN: M 或 N 维度索引
                     * - K: K 维度索引
                     * - SF_K: Scale Factor K 维度索引 (FP8 专用，BF16 不使用)
                     */
                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];

                    const auto m_idx = scheduler.template get_global_idx<kWithGroupOffsetA, IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                    const auto n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);

                    // ========== K 维度索引计算 ==========
                    /**
                     * @brief 根据布局计算 K 维度索引
                     * 
                     * **布局依赖**:
                     * - kMajorA == K: A 是列主序，K 维度连续 → 不需要 group offset
                     * - kMajorA == MN: A 是行主序，K 维度不连续 → 需要 group offset
                     */
                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), IndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), IndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    // ========== Batched GEMM 配置 ==========
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                    
                    // ========== 发起 TMA 传输 ==========
                    /**
                     * @brief 使用 TMA 从 Global Memory 异步加载 A/B 到 Shared Memory
                     * 
                     * **TMA 拷贝模式**:
                     * - 2D: (m_idx, k_idx) 或 (n_idx, k_idx)
                     * - 3D (Batched): (m_idx, k_idx, batch_idx)
                     * 
                     * **参数顺序**:
                     * tma_copy<BLOCK_MAJOR, BLOCK_MINOR, SWIZZLE_MODE, DTYPE, IS_BATCHED>(
                     *     tensor_map,      // TMA 描述符
                     *     full_barrier,    // 完成屏障
                     *     smem_ptr,        // Shared memory 目标地址
                     *     idx_major,       // 主要维度索引
                     *     idx_minor,       // 次要维度索引
                     *     num_multicast,   // Multicast 数量
                     *     batch_idx        // Batch 索引 (仅 3D)
                     * )
                     */
                    if constexpr (kMajorA == cute::UMMA::Major::K)
                        tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], k_a_idx, m_idx, num_tma_multicast_a, batch_idx);
                    if constexpr (kMajorA == cute::UMMA::Major::MN)
                        tma_copy<BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], m_idx, k_a_idx, num_tma_multicast_a, batch_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::K)
                        tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], k_b_idx, n_idx, num_tma_multicast_b, batch_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::MN)
                        tma_copy<BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], n_idx, k_b_idx, num_tma_multicast_b, batch_idx);

                    // ========== 通知消费者数据已就绪 ==========
                    /**
                     * @brief Barrier arrive: 数据加载完成
                     * expect_tx: 预期传输的字节数 (A + B)
                     * 这允许 TMA 异步传输，Math warp 可以提前准备
                     */
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
            }

            // ========== 清理阶段 ==========
            /**
             * @brief 等待所有 empty barriers，确保安全地销毁分布式 barrier
             * 仅在使用 Multicast 时需要，因为涉及多个 CTAs 的同步
             */
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    // ========== Math Warp-Groups (Consumer) ==========
    /**
     * @brief Math warp-groups 负责执行 WGMMA 计算和结果写回
     * 
     * **执行路径**:
     * 1. 分配更多寄存器 (224-248 个)，用于 WGMMA 累加器
     * 2. 使用 __shfl_sync 统一读取 warp group 索引
     * 3. 每个 warp group 独立处理分配的 blocks
     * 4. 支持 merged stages (仅 Normal GEMM + NT 布局)
     */
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // 使用 __shfl_sync 统一读取 math_wg_idx
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        
        // ========== Merged Stages 配置 ==========
        /**
         * @brief Merged stages 仅发生在 NT 布局的 Normal GEMM
         * 
         * **目的**: 减少 warpgroup_wait 的同步开销
         * **方法**: 将多个小 stage 合并成一个大 stage
         * 
         * 示例:
         * - BLOCK_K = 128 (合并后)
         * - kNumStagesPerMerge = 2
         * - BLOCK_ATOM_K = 64 (实际 WGMMA 使用的 K 维度)
         */
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        
        // ========== GMMA 描述符初始化 ==========
        /**
         * @brief 创建 GMMA (Global Memory Matrix Multiply) 描述符
         * 
         * **描述符作用**: 描述 Shared Memory 中矩阵的布局，供 WGMMA 使用
         * 
         * **参数**:
         * - smem_ptr: Shared memory 基地址
         * - wg_idx * M: Warp group 偏移 (每个 warp group 处理不同的 M 块)
         * - 0: K 维度偏移 (初始为 0)
         * 
         * **返回值**:
         * - reg32_[0]: 描述符的低 32 位 (后续通过修改它来更新描述符)
         */
        auto a_desc = make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = make_gmma_desc<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // ========== Block 主循环 ==========
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // ========== WGMMA 配置 ==========
            /**
             * @brief 配置 WGMMA wave 和累加器
             * 
             * **WAVE_BLOCK_M**: 每个 WGMMA wave 处理的 M 维度
             * - BLOCK_M <= WGMMA::M: 一个 wave 处理完
             * - BLOCK_M > WGMMA::M: 需要多个 waves (最多 2 个)
             * 
             * **累加器大小**:
             * - WGMMA::kNumAccum: 每个 WGMMA 指令的累加器数量
             * - BLOCK_M / WAVE_BLOCK_M: waves 数量
             * - 总累加器 = WGMMA::kNumAccum × (BLOCK_M / WAVE_BLOCK_M)
             */
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};  // 初始化为 0

            // ========== WGMMA Store 线程选择 ==========
            /**
             * @brief 选择哪些线程负责将 WGMMA 结果存回 Shared Memory
             * 
             * **规则**:
             * - BLOCK_M >= 64: 所有线程都参与 (需要多个 warp groups)
             * - BLOCK_M < 64: 仅部分线程参与 (一个 warp group 足够)
             * 
             * **kNumWGMMAStoreThreads**: 需要的线程数
             * = WAVE_BLOCK_M × (128 / WGMMA::M)
             */
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;

            // ========== Empty Barrier Arrive ==========
            /**
             * @brief 通知 TMA 生产者：当前 stage 已使用完，可以重用
             * 
             * **Multicast 模式**:
             * - kNumTMAMulticast == 1: 简单 arrive
             * - kNumTMAMulticast > 1: 需要指定目标 CTA
             *   - scheduler.is_peer_cta_alive: 检查相邻 CTA 是否存活
             *   - 如果存活，发送到相邻 CTA
             *   - 否则，发送到当前 CTA (避免无效 multicast)
             */
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // ========== K 维度循环 (WGMMA 计算) ==========
            // TODO: remove some useless computation for unaligned Ms
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // ========== 更新 GMMA 描述符 ==========
                /**
                 * @brief 计算当前 stage 的 GMMA 描述符基地址
                 * 
                 * **描述符更新公式**:
                 * - base_lo + stage_idx × (SMEM_SIZE / 16)
                 * - /16: 因为描述符使用 16-byte 对齐的地址
                 * 
                 * **为什么只更新 base_lo?**
                 * - 描述符的其他字段 (布局、swizzle 等) 不变
                 * - 只需修改基地址即可指向不同的 stage
                 */
                const auto& a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto& b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                // ========== 等待 TMA 数据到达 ==========
                /**
                 * @brief 等待当前 stage 的数据加载完成
                 * phase: 当前相位 (与 TMA 生产者同步)
                 */
                full_barriers[stage_idx]->wait(phase);

                // ========== WGMMA 指令提交 ==========
                /**
                 * @brief 提交 WGMMA 指令序列
                 * 
                 * **执行流程**:
                 * 1. warpgroup_fence_operand: 标记累加器为"正在使用"
                 * 2. warpgroup_arrive: 开始一批 WGMMA 指令
                 * 3. 执行所有 WGMMA 指令 (嵌套循环)
                 * 4. warpgroup_commit_batch: 提交整批指令
                 * 5. warpgroup_fence_operand: 标记累加器为"完成"
                 * 6. warpgroup_wait<0>: 等待所有 WGMMA 完成
                 * 
                 * **为什么需要 fence 和 wait?**
                 * - WGMMA 是异步执行的
                 * - fence 防止指令重排序
                 * - wait 确保所有累加器就绪后才能进行 epilogue
                 */
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                // ========== WGMMA 核心计算 ==========
                /**
                 * @brief 执行 WGMMA 矩阵乘法
                 * 
                 * **嵌套循环结构**:
                 * 1. local_idx: 遍历 waves (BLOCK_M / WAVE_BLOCK_M)
                 * 2. k: 遍历 K 维度 (BLOCK_K / WGMMA::K)
                 * 
                 * **每次迭代**:
                 * - 更新 A/B 描述符 (指向正确的 K 块)
                 * - 调用 WGMMA::wgmma 执行矩阵乘法
                 * - 结果累加到 shifted_accum
                 * 
                 * **描述符更新**:
                 * - advance_gmma_desc_lo: 修改描述符的 low 32 位
                 * - 参数：base_lo, M_offset, K_offset, stride
                 */
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        const uint32_t& atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K;
                        a_desc.reg32_[0] = advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, nv_bfloat16>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                        b_desc.reg32_[0] = advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, nv_bfloat16>(
                            b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                // 提交 WGMMA 指令批处理并等待完成
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();  // 等待所有 WGMMA 完成

                // ========== 通知 TMA 缓冲区已释放 ==========
                empty_barrier_arrive(stage_idx);
            }

            // ========== Epilogue: 结果写回 ==========
            /**
             * @brief TMA 检查：配置 TMA store 参数
             * 
             * **关键参数**:
             * - kNumElemBytes: BF16 = 2 bytes
             * - TMA_D_BLOCK_N: TMA store 的 N 维度分块大小
             *   - kSwizzleDMode=0: 使用 BLOCK_N
             *   - kSwizzleDMode>0: 使用 swizzle 宽度
             * - WGMMA_M_PER_WARP: 每个 warp 处理的 M 维度 (WGMMA::M / 4)
             */
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // 跳过无效的 WGMMA store (针对未填满的部分)
            if (not do_wgmma_store)
                continue;

            // ========== 等待 TMA Store 完成 ==========
            /**
             * @brief 等待上一轮 TMA store 完成，避免数据竞争
             * tma_store_wait<0>: 等待所有 pending 的 TMA stores
             * NamedBarrier::sync: 同步所有参与 store 的线程
             */
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            // ========== 写回 Shared Memory (BF16 输出) ==========
            /**
             * @brief 使用 STSM 指令将累加器结果写回 Shared Memory
             * 
             * **STSM (Shared Memory Tensor Store Matrix Multiply)**:
             * - 专为 WGMMA 结果存储优化的指令
             * - 支持 BF16 打包存储
             * - 需要 swizzling 避免 bank conflicts
             * 
             * **Swizzling 逻辑** (kSwizzleDMode > 0):
             * 1. 计算 atom 偏移和 atom 内偏移
             * 2. 根据 bank group 索引进行 XOR 交织
             * 3. 重塑 atom 视图：(BLOCK_M, kSwizzleDMode/16) → (BLOCK_M*kSwizzleDMode/16/8, 8)
             * 4. 应用 XOR swizzle: col ^= row % (kSwizzleDMode/16)
             * 
             * **向量存储**:
             * - 每次存储 4 个累加器值 (打包成 2 个 BF16x2)
             * - 使用 SM90_U32x2_STSM_N 指令
             */
            if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                // Write back to shared memory using STSM and issue TMA stores
                DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
                DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        // 计算 swizzling atom 偏移和 atom 内偏移
                        uint8_t* smem_ptr = nullptr;
                        if constexpr (kSwizzleDMode > 0) {
                            // 计算 bank group 字节数
                            constexpr uint32_t kNumBankGroupBytes = 16;
                            auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                            // 计算 atom 中要写入的 bank group 索引
                            auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                            // 重塑 atom 视图并应用 swizzle
                            //  原始视图：(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)
                            //  新视图：(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)
                            constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                            col ^= row % (kSwizzleDMode / 16);  // XOR swizzle

                            // 加回到基地址
                            // 注意：修改前请三思，因为这可能影响指令数量
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // 基地址
                                warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp 偏移
                                m_offset * kSwizzleDMode +                                 // Wave 偏移
                                atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom 偏移 (常量)
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // Atom 内偏移
                        } else {
                            // 无 swizzling
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                        }

                        // 注意：只有 16 个 lane 的地址会被使用
                        SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                            smem_ptr
                        );
                    }
                }
            // ========== 写回 Shared Memory (Float 输出) ==========
            /**
             * @brief 使用 st.shared 将累加器结果写回 (当输出为 float 时)
             * 
             * **为什么不用 STSM?**
             * - STSM 仅支持 BF16/FP16 等压缩格式
             * - Float 输出需要使用常规的 st.shared 指令
             * 
             * **存储模式**:
             * - 每个线程存储 2 个 float2 (8 个 float)
             * - smem_d_0: 前 8 个累加器
             * - smem_d_1: 后 8 个累加器 (偏移 8 行)
             */
            } else {
                // 当 STSM 不可用时使用 st.shared
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    auto smem_d_0 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
                    auto smem_d_1 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        st_shared(smem_d_0 + i * 4, make_float2(shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]));
                        st_shared(smem_d_1 + i * 4, make_float2(shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]));
                    }
                }
            }
            
            // ========== TMA Store 写回 Global Memory ==========
            /**
             * @brief 使用 TMA 将 Shared Memory 结果异步写回 Global Memory
             * 
             * **执行流程**:
             * 1. tma_store_fence: 发起 TMA store fence (确保所有 stores 可见)
             * 2. NamedBarrier::sync: 同步所有参与 store 的线程
             * 3. 发起 TMA store 异步传输
             * 
             * **TMA Store 模式**:
             * - kWithAccumulation=true: SM90_TMA_REDUCE_ADD_2D (累加模式)
             * - kWithAccumulation=false: SM90_TMA_STORE_2D (覆盖模式)
             * - Batched: SM90_TMA_STORE_3D (3D 张量)
             */
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            // 使用 TMA store 写回 Global Memory
            const auto m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                
                // ========== Batched GEMM (3D TMA Store) ==========
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_cd, smem_ptr,
                                                  n_block_idx * BLOCK_N + in_block_n_offset,
                                                  m_idx, scheduler.current_group_idx);
                } else {
                    // ========== 2D TMA Store ==========
                    /**
                     * @brief 根据 kWithAccumulation 选择 TMA store 模式
                     * 
                     * **累加模式** (kWithAccumulation=true):
                     * - SM90_TMA_REDUCE_ADD_2D: 读取 + 累加 + 写回
                     * - 用于需要累加的场景 (如 Psum layout)
                     * 
                     * **覆盖模式** (kWithAccumulation=false):
                     * - SM90_TMA_STORE_2D: 直接覆盖
                     * - 用于标准 GEMM
                     */
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr,
                                     n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                }
                // 异步发起 TMA store
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    // ========== 架构检查 ==========
    /**
     * @brief 仅支持 SM90a (Hopper 架构)
     * 如果在不支持的架构上运行，触发断言
     */
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
