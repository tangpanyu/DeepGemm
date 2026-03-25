/**
 * @file sm90_fp8_gemm_1d2d.cuh
 * @brief SM90 (Hopper 架构) FP8 GEMM 核心实现 - 1D2D Tiling 版本
 * 
 * **1D2D Tiling 含义**:
 * - "1D": K 维度分块 (BLOCK_K = 128)
 * - "2D": M 和 N 维度也进行分块，多个 blocks 协作处理大矩阵
 * - 相比 1D1D: 支持更大的矩阵规模，更高的并行度
 * 
 * **计算任务**: D = A @ B.T (FP8 输入，BF16 输出)
 * - A: [M, K] FP8 (E4M3)
 * - B: [N, K] FP8 (E4M3)
 * - D: [M, N] BF16 (输出)
 * - Scale factors: per-128-channel 的 FP32 缩放因子
 * 
 * **关键特性**:
 * - 使用 Hopper TMA 硬件单元
 * - WGMMA (Warp Group MMA) 指令
 * - 多级软件流水线
 * - TMA Multicast (最多 2 CTAs)
 * - Shared memory swizzling (避免 bank conflicts)
 * - 动态 scaling factor 加载 (适应边界情况)
 */
#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

/**
 * @brief 运行时 dispatch 迭代次数的辅助函数
 * 
 * **作用**: 在编译期展开循环，优化性能
 * 用于处理 BLOCK_N 不能被 BLOCK_K 整除的情况
 * 
 * @tparam kNumFormerIters - 当前迭代的次数
 * @tparam kGap - 每次递增的步长
 * @tparam kEnd - 最大迭代次数
 * @tparam func_t - 函子类型
 * @param num_former_iters - 实际的迭代次数
 * @param func - 要执行的函子
 */
template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
__device__ void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

/**
 * @brief FP8 GEMM 核心 kernel 函数模板 (1D2D tiling)
 * 
 * **模板参数详解**:
 * 
 * @tparam kMajorSFB - Scale Factor B 的内存布局
 *         - MN: row-major (B 与 D 同布局)
 *         - K: col-major (B 与 K 同布局)
 * 
 * @tparam SHAPE_M, N, K - 全局矩阵维度，编译期常量 (0 表示运行时动态)
 * 
 * @tparam kNumGroups - MoE 分组数量
 *         - Normal/Batched GEMM: = 1
 *         - M-grouped GEMM: > 1
 * 
 * @tparam BLOCK_M, N, K - Thread block 级别的分块大小
 *         - BLOCK_K = 128 (固定)
 *         - BLOCK_M, N: 通常 64/128/256
 *         - 2D tiling: 一个 block 只处理 BLOCK_M x BLOCK_N 的一小块
 * 
 * @tparam kSwizzleAMode, BMode, DMode - Shared memory 交织模式
 *         - A/B: 避免 load 时的 bank conflicts
 *         - D: 避免 store 时的 bank conflicts
 *         - 典型值：0 (无), 32, 64, 128 (字节)
 * 
 * @tparam kNumStages - 主流水线的级数
 *         - 通常 4-8 级
 * 
 * @tparam kNumLastStages - 最后阶段的特殊处理 (未使用)
 * 
 * @tparam kNumTMAThreads - TMA warp 的线程数 (固定 128)
 * 
 * @tparam kNumMathThreads - Math warp 组的总线程数
 *         - 128 或 256
 * 
 * @tparam kNumTMAMulticast - TMA 多播的 CTA 数量 (1 或 2)
 * 
 * @tparam kIsTMAMulticastOnA - TMA 多播的方向
 *         - true: M 方向 (A 矩阵)
 *         - false: N 方向 (B 矩阵)
 * 
 * @tparam kNumSMs - GPU 的 SM 数量
 * 
 * @tparam kGemmType - GEMM 类型
 *         - Normal: 标准矩阵乘法
 *         - MGroupedMasked: M 分组掩码布局
 *         - Batched: Batched GEMM
 * 
 * @tparam epilogue_type_t - Epilogue 类型 (索引变换)
 */
template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages, uint32_t kNumLastStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d2d_impl(
    // ========== Global Memory 输入指针 ==========
    float* sfb,                   // Scale Factor B 的全局内存指针 [shape_n_sfb, shape_k_scales]
    int* grouped_layout,          // 分组布局数组 (M-grouped/batched 时使用)
    
    // ========== 运行时矩阵维度 ==========
    uint32_t shape_m,             // M 维度大小 (行数)
    uint32_t shape_n,             // N 维度大小 (列数)
    uint32_t shape_k,             // K 维度大小 (收缩维度)
    
    // ========== TMA 描述符 (grid constant) ==========
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,       // A 矩阵的 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,       // B 矩阵的 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_d,       // D 矩阵的 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_sfa      // Scale Factor A 的 TMA 描述符
){
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // ========== 编译期常量检查 ==========
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    // 检查 BLOCK_N 和 BLOCK_K 的关系，确保 scaling factor 不会过多
    DG_STATIC_ASSERT(constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or (constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // ========== WGMMA 指令配置 ==========
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // ========== 编译期形状覆盖 ==========
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // ========== Shared Memory 布局设计 ==========
    // 判断是否需要统一的 scale B (BLOCK_K 是 BLOCK_N 的整数倍时)
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    
    // D 缓冲区：BF16 输出，需要 1024B 对齐
    static constexpr uint32_t SMEM_D_SIZE = constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    
    // A/B 缓冲区：FP8 数据，kNumStages 级流水线
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    
    // Scale Factor A 缓冲区：FP32，每个 BLOCK_M 一个 scale
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    
    // Scale Factor B 的特殊处理:
    // - shape_k_scales: K 维度的 scale 数量
    // - shape_n_sfb: N 维度的 scale 块数量
    const uint32_t& shape_k_scales = ceil_div(shape_k, BLOCK_K);
    const uint32_t& shape_n_sfb = ceil_div(shape_n, BLOCK_K);
    // SFB 大小：根据是否统一 scale 决定存储 1 行还是 2 行
    const uint32_t& smem_sfb_size = align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier));

    // ========== WGMMA 内存边界检查 ==========
    // 确保 WGMMA 需要的 A 矩阵不超过 shared memory 容量
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // ========== Thread 角色分配 ==========
    const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);  // K 维度总块数
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // ========== TMA Descriptor 预取 ==========
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // ========== Shared Memory 声明与分区 ==========
    // extern __shared__: 动态大小的 shared memory
    // __align__(1024): 1024B 对齐，优化 swizzle 访问
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // ========== Shared Memory 指针布局 ==========
    // D 输出缓冲区：BF16 类型
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    
    // A 数据缓冲区：kNumStages 级流水线
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    
    // B 数据缓冲区：kNumStages 级流水线
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    
    // Scale Factor A 缓冲区
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    
    // Scale Factor B 缓冲区：连续存储，非多级缓冲
    // 存储所有 K-scales 和可能的第二行 scales
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // ========== Barrier 同步原语 ==========
    // Barrier 起始指针：紧跟在 smem_sfb 之后
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    
    // full_barriers[i]: 第 i 级流水线的"数据就绪"屏障
    auto full_barriers     = PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    
    // empty_barriers[i]: 第 i 级流水线的"缓冲区空"屏障
    auto empty_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // ========== Barrier 初始化 ==========
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // 初始化所有流水线级的 barrier
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // full_barrier: 等待 1 个 TMA warp arrive
            full_barriers[i]->init(1);
            // empty_barrier: 等待所有 math warps arrive
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // fence_barrier_init(): Hopper 专用指令，确保 barrier 可见
        cutlass::arch::fence_barrier_init();
    }

    // ========== 全局同步 ==========
    // Cluster sync (multicast > 1) 或 block sync
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // ========== 寄存器重配置 ==========
    constexpr uint32_t kNumTMARegisters = 40;  // TMA warp 使用较少寄存器
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;  // Math warp 使用较多寄存器

    // ========== Block Scheduler 初始化 ==========
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // ========== 流水线相位管理 ==========
    uint32_t stage_idx = 0, phase = 0;
    // advance_pipeline: 推进到下一级流水线
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        // 完成一轮后翻转相位
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ========== TMA Warp 路径：数据加载 ==========
    if (warp_idx >= kNumMathThreads / 32) {
        // 释放寄存器给 TMA 操作
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // 只有一个 warp 实际执行 (warp_idx == kNumMathThreads/32 + 2)
        // 使用第 3 个 warp，避免与 WGMMA (BLOCK_M==32) 冲突
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // ========== TMA Warp 持久化调度主循环 ==========
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // ========== TMA Multicast 配置 ==========
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                // ========== K 维度软件流水线主循环 ==========
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // [步骤 1] 等待消费者释放缓冲区
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // [步骤 2] 发起 TMA 拷贝 A 和 Scale Factor A
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                    // 是否有分组偏移 (M-grouped masked)
                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    
                    // TMA 拷贝 A: [BLOCK_M, BLOCK_K]
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                             num_tma_multicast_a, batch_idx);
                    
                    // TMA 拷贝 Scale Factor A: [BLOCK_M, 1]
                    tma_copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                             smem_sfa[stage_idx], m_block_idx * BLOCK_M, scheduler.template get_global_idx<kWithGroupOffsetA, IndexType::SF_K>(shape_k_scales, 1, k_block_idx),
                             num_tma_multicast_a);

                    // [步骤 3] 发起 TMA 拷贝 B: [BLOCK_N, BLOCK_K]
                    // 注意：B 的 scale factor 不在这里加载，由 math warp 提前加载
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                             smem_b[stage_idx], k_idx, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b, batch_idx);
                    
                    // [步骤 4] 通知 barrier：预期传输的字节数
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                }
            }

            // ========== 清理分布式 Barrier ==========
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // ========== Math Warp 路径：WGMMA 计算 ==========
        // 分配更多寄存器用于存放 WGMMA 累加器
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // ========== Thread 坐标计算 ==========
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);  // Math warp 组索引
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;  // 两行的起始索引

        // ========== Shared Memory Descriptor 预计算 ==========
        // 预先计算 A/B 的 shared memory 描述符基地址
        auto a_desc = make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        auto b_desc = make_smem_desc(smem_b[0], 1);
        // 通过 shuffle 共享描述符的低 32 位 (避免重复计算)
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // ========== Math Warp 持久化调度主循环 ==========
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // ========== 动态计算 B Scale 加载数量 ==========
            // 处理边界情况：当 BLOCK_N 不能被 BLOCK_K 整除时
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                // 非统一 scale: 需要加载两行 scales
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            // 决定加载 1 行还是 2 行 scales
            uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

            // ========== 提前加载 B Scales (与 TMA Store 重叠) ==========
            // 使用 math warp groups 加载 B 的 scaling factors
            // 除了第一个 warp 外，其他 warps 可以提前加载下一轮的 scales
            if (threadIdx.x >= 32) {
                // 计算前一组的偏移
                auto previous_group_offset = scheduler.template get_global_idx<true, IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                // 根据布局计算 stride
                const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
                auto local_sfb = sfb + previous_group_offset + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;

                // 分布式加载 scales 到 shared memory
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                    st_shared(smem_sfb + i, __ldg(i < shape_k_scales ? local_sfb + i * stride_k_sfb : local_sfb + (i - shape_k_scales) * stride_k_sfb + stride_n_sfb));
            }
            // 同步所有 math threads
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            // ========== WGMMA 累加器配置 ==========
            // WAVE_BLOCK_M: 每个 wave 处理的 M 维度大小
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            
            // 累加器数组
            float accum[WGMMA::kNumAccum];  // WGMMA 原始结果
            float final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};  // 反量化后的累加结果
            
            // ========== WGMMA Store Thread 分配 ==========
            // 决定哪些 threads 负责将 WGMMA 结果写入 shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // ========== Empty Barrier 到达函子 ==========
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    // 无多播：lane_idx=0 代表整个 warp arrive
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    // 有多播：指定目标 CTA
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // ========== 跳过无效计算 ==========
            // 检查当前 block 是否在有效范围内
            if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
                // ========== 编译期优化参数 ==========
                // 当 BLOCK_K/BLOCK_N 较小时，需要特殊优化
                constexpr bool kShouldOptimize = BLOCK_K / constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // ========== Dispatch + WGMMA 主循环 ==========
                // 运行时 dispatch num_former_iters，展开循环
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        // 更新 shared memory descriptor 的偏移
                        const auto& a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto& b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        // [步骤 1] 读取 B Scales
                        float scale_b_0 = ld_shared(smem_sfb + k_block_idx), scale_b_1;
                        // 非统一 scale 时，需要读取第二行
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ld_shared(smem_sfb + k_block_idx + shape_k_scales);

                        // [步骤 2] 等待 TMA 数据就绪
                        full_barriers[stage_idx]->wait(phase);

                        // [步骤 3] Wave 循环处理 M 维度
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // [步骤 3a] 读取 A Scales (必须在 warpgroup_arrive 之前)
                            auto scale_a_0 = do_wgmma_store ? ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
                            auto scale_a_1 = do_wgmma_store ? ld_shared(smem_sfa[stage_idx] + r_1 + m_offset) : 0;

                            // [步骤 3b] WGMMA 矩阵乘法
                            // Fence: 防止重排序
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                warpgroup_fence_operand(accum[i]);
                            
                            // Arrive: 通知 warp 组开始 MMA
                            warpgroup_arrive();
                            
                            // K 循环：BLOCK_K / WGMMA::K 次迭代
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                // 更新 descriptor 偏移
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                // 发起 WGMMA 指令
                                WGMMA::wgmma(a_desc, b_desc, accum, k);
                            }
                            
                            // Commit: 提交所有 WGMMA 指令
                            warpgroup_commit_batch();
                            
                            // Fence: 确保累加器安全
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                warpgroup_fence_operand(accum[i]);
                            
                            // Wait: 等待所有操作完成
                            warpgroup_wait<0>();

                            // [步骤 3c] 最后一个 wave 通知 empty barrier
                            if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                                empty_barrier_arrive();

                            // 跳过无效部分的 promotion
                            if (not do_wgmma_store)
                                continue;

                            // [步骤 3d] FP8 反量化 + 累加
                            // 预计算 scale 乘积
                            float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                            float scale_0_1, scale_1_1;
                            if constexpr (not kMustUseUniformedScaleB)
                                scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // 根据是否统一 scale 选择正确的乘积
                                const bool& predicate = kMustUseUniformedScaleB or i < num_former_iters;
                                shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                            }
                        }
                    }
                });
            } else {
                // ========== 跳过无效计算 ==========
                // 如果当前 block 不在有效范围内，只需等待 barrier
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // ========== TMA Store 配置检查 ==========
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            // TMA D 的 N 维度块大小：根据 swizzle mode 决定
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            
            // 编译期检查
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // 跳过无效的 WGMMA store
            if (not do_wgmma_store)
                continue;

            // ========== 等待之前的 TMA Store 完成 ==========
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // ========== STSM + TMA Store Epilogue ==========
            // 使用 STSM 指令写入 shared memory，然后发起 TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // ========== Shared Memory Swizzling ==========
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // 计算 swizzling atom 偏移和 atom 内偏移
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // 计算 bank group 索引
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Swizzle 变换:
                        // 原始视图：(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)
                        // 新视图：(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);  // XOR swizzle

                        // 计算最终的 shared memory 指针
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // 无 swizzling，直接 padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // ========== STSM 指令写入 ==========
                    // 只有 16 个 lanes 的地址被使用
                    SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            // TMA Store Fence
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // ========== TMA Store 写回全局内存 ==========
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                // 计算 block 内的 N 偏移
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                
                // 应用 epilogue 索引变换
                auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                
                // 根据 GEMM 类型选择 2D 或 3D TMA store
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                  n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
