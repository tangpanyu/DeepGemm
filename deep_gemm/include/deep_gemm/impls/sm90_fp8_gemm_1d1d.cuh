/**
 * @file sm90_fp8_gemm_1d1d.cuh
 * @brief SM90 (Hopper 架构) FP8 GEMM 核心实现 - 1D Tiling 版本
 * 
 * **1D Tiling 含义**:
 * - 只在 K 维度上进行分块 (BLOCK_K = 128)
 * - M 和 N 维度由单个 thread block 完整处理
 * - 适合中等规模矩阵，平衡寄存器和 shared memory 使用
 * 
 * **计算任务**: D = A @ B.T (FP8 输入，FP32 输出)
 * - A: [M, K] FP8 (E4M3)
 * - B: [N, K] FP8 (E4M3) 
 * - D: [M, N] FP32 (累加结果)
 * - Scale factors: per-128-channel 的 FP32 缩放因子
 * 
 * **关键特性**:
 * - 使用 Hopper TMA (Tensor Memory Accelerator) 硬件单元
 * - WGMMA (Warp Group Matrix Multiply Accumulate) 指令
 * - 多级软件流水线隐藏内存延迟
 * - TMA Multicast: 多个 CTAs 共享一次数据加载
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

/**
 * @brief FP8 GEMM 核心 kernel 函数模板 (1D tiling)
 * 
 * **模板参数详解**:
 * 
 * @tparam SHAPE_M, N, K - 全局矩阵维度，编译期常量 (0 表示运行时动态)
 *         - 如果非零，编译器可以优化掉一些运行时检查
 *         - 如果为 0，使用运行时传入的 shape_m/n_k
 * 
 * @tparam kNumGroups - MoE (Mixture of Experts) 分组数量
 *         - Normal GEMM: = 1
 *         - K-grouped GEMM: > 1 (多个 K 维度不同的专家网络)
 * 
 * @tparam BLOCK_M, N, K - Thread block 级别的矩阵分块大小
 *         - BLOCK_K = 128 (固定，FP8 scaling 的粒度)
 *         - BLOCK_M, N: 通常为 64/128/256
 *         - 1D tiling: 一个 block 处理 BLOCK_M x BLOCK_N 的输出块
 * 
 * @tparam kSwizzleAMode, BMode - Shared memory 交织模式 (bank conflict 优化)
 *         - 通过 XOR 变换打散内存访问模式，避免 shared memory bank 冲突
 *         - 典型值：0 (无), 32, 64, 128 (字节)
 * 
 * @tparam kNumStages - 软件流水线级数
 *         - 通常 4-8 级，重叠 TMA 加载和 MMA 计算
 *         - 越多越能隐藏延迟，但消耗更多 shared memory
 * 
 * @tparam kNumTMAThreads - TMA warp 的线程数 (固定 128 = 4 warps)
 *         - 专职负责从 global memory 搬运数据到 shared memory
 * 
 * @tparam kNumMathThreads - Math warp 组的线程总数
 *         - 专职负责 WGMMA 计算，必须是 128 的倍数
 *         - 例如：256 = 2 个 warp 组，512 = 4 个 warp 组
 * 
 * @tparam kNumTMAMulticast - TMA 多播的 CTA 数量 (1 或 2)
 *         - 1: 无多播，每个 CTA 独立加载数据
 *         - 2: 2 个 CTAs 共享一次 TMA 加载，减少 L2 带宽压力
 * 
 * @tparam kIsTMAMulticastOnA - TMA 多播的方向
 *         - true: 在 M 维度多播 (A 矩阵方向)
 *         - false: 在 N 维度多播 (B 矩阵方向)
 * 
 * @tparam kNumSMs - 目标 GPU 的 SM 数量
 *         - 用于 scheduler 决定如何分配 blocks 到 SMs
 * 
 * @tparam kGemmType - GEMM 类型枚举
 *         - Normal: 标准矩阵乘法
 *         - KGroupedContiguous: K 维度的分组 GEMM (MoE 场景)
 * 
 * @tparam cd_dtype_t - C/D 矩阵的数据类型 (必须是 float)
 */
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_impl(
    // ========== Global Memory 输入指针 ==========
    __nv_fp8_e4m3* gmem_a_ptr,      // A 矩阵全局内存指针 [M, K]
    __nv_fp8_e4m3* gmem_b_ptr,      // B 矩阵全局内存指针 [N, K]
    int* grouped_layout,            // K-grouped GEMM 的布局数组 (记录每组的实际 K 大小)
    cute::TmaDescriptor* tensor_map_buffer,  // TMA 描述符缓冲区 (用于动态修改)
    // ========== 运行时矩阵维度 ==========
    uint32_t shape_m,               // M 维度大小 (行数)
    uint32_t shape_n,               // N 维度大小 (列数)
    uint32_t shape_k,               // K 维度大小 (收缩维度)
    // ========== TMA 描述符 (grid constant) ==========
    // __grid_constant__: 在整个 grid 中所有 threads 看到相同的值
    const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,     // A 矩阵的 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,     // B 矩阵的 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,        // Scale Factor A 的 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,        // Scale Factor B 的 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_cd          // C/D 矩阵的 TMA 描述符
){
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // ========== 编译期常量检查 ==========
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous, "Invalid GEMM type");

    // ========== WGMMA 指令配置 ==========
    // 根据 BLOCK_N 选择合适的 WGMMA 指令变体
    // WGMMA 是 Hopper 的 Warp Group MMA 指令，多个 warps 协作完成矩阵乘法
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;  // Cluster 事务屏障
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // ========== 编译期形状覆盖 ==========
    // 如果模板参数非零，优先使用编译期常量 (编译器可以优化)
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // ========== Shared Memory 布局设计 ==========
    // Shared memory 总览:
    // +-----------------------+ <- 0x0
    // | Tensor Map Descriptors|   (仅 K-grouped 需要，4 个描述符 * 64B)
    // +-----------------------+ <- SMEM_TENSOR_MAP_SIZE
    // | D Accumulator Buffer  |   BLOCK_M * BLOCK_N * 4B (FP32)
    // +-----------------------+ <- SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE
    // | A Buffer (Stage 0)    |   BLOCK_M * BLOCK_K * 1B (FP8)
    // | B Buffer (Stage 0)    |   BLOCK_N * BLOCK_K * 1B (FP8)
    // | ...                   |   kNumStages 级流水线
    // +-----------------------+ <- + kNumStages * (SMEM_A + SMEM_B)
    // | Scale Factor A Buffer |   BLOCK_M * 4B (FP32)
    // | Scale Factor B Buffer |   BLOCK_N * 4B (FP32)
    // | ...                   |   kNumStages 级
    // +-----------------------+ <- + kNumStages * (SMEM_SFA + SMEM_SFB_aligned)
    // | Barriers              |   2 * kNumStages 个屏障对象
    // +-----------------------+ <- 结束
    
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 4 : 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);  // 输出累加缓冲区
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);  // A 每阶段
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);  // B 每阶段
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);  // Scale A 每阶段
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);  // Scale B 每阶段
    // 128B 对齐：保证 TMA 访问效率，避免 bank conflict
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u);
    DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment");

    // ========== Thread 角色分配 ==========
    // Hopper 架构中，threadIdx.x 被分配到不同的 warp groups
    // warp_idx: 当前 thread 所属的 warp 编号 (0-based)
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    // lane_idx: warp 内的线程编号 (0-31)
    const uint32_t lane_idx = threadIdx.x % 32;

    // ========== TMA Descriptor 预取 ==========
    // 提前将 TMA 描述符加载到 texture cache，减少首次访问延迟
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();  // 等待所有 threads 完成预取

    // ========== Shared Memory 声明与分区 ==========
    // extern __shared__: 动态大小的 shared memory (由 kernel launch 时指定)
    // __align__(1024): 1024B 对齐，优化 swizzle-128B 访问模式
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // ========== PatternVisitor: Shared Memory 地址计算器 ==========
    // PatternVisitor 是一个函数对象，根据索引 i 返回对应的指针
    // 类似 lambda 数组访问：smem_tensor_map_a[i] 自动计算偏移
    
    // Tensor Map 描述符在 shared memory 中的位置 (用于 K-grouped 动态修改)
    auto smem_tensor_map_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + static_cast<uint32_t>(sizeof(cute::TmaDescriptor)) * i);
    });
    auto smem_tensor_map_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + static_cast<uint32_t>(sizeof(cute::TmaDescriptor)) * (2 + i));
    });
    // Global memory 中的 tensor map 缓冲区 (每个 block 有独立的 4 个描述符副本)
    auto gmem_tensor_map_a = PatternVisitor([=](const uint32_t& i) { return tensor_map_buffer + blockIdx.x * 4 + i; });
    auto gmem_tensor_map_b = PatternVisitor([=](const uint32_t& i) { return tensor_map_buffer + blockIdx.x * 4 + 2 + i; });

    // ========== Shared Memory 数据区指针 ==========
    // D 累加器缓冲区：存储 WGMMA 的 FP32 中间结果
    auto smem_d = reinterpret_cast<float*>(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    
    // A/B 数据缓冲区：kNumStages 级流水线，每级一个 BLOCK
    // PatternVisitor 自动计算偏移：base_offset + i * stage_size
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE)); 
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });
    
    // Scaling factors 缓冲区：FP32 格式，每个 BLOCK_M/BLOCK_N 一个 scale
    constexpr auto SMEM_SF_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + i * SMEM_SFA_SIZE_PER_STAGE));
    });
    auto smem_sfb = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE));
    });

    // ========== Barrier 同步原语 ==========
    // full_barriers[i]: 第 i 级流水线的"数据就绪"屏障
    //   - TMA 加载完成后 arrive，表示"缓冲区已满"
    //   - Math warp 等待这个屏障才能开始计算
    constexpr auto SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    auto full_barriers = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + i * static_cast<uint32_t>(sizeof(Barrier))));
    });
    
    // empty_barriers[i]: 第 i 级流水线的"缓冲区空"屏障
    //   - Math warp 计算完成后 arrive，表示"缓冲区已空"
    //   - TMA warp 等待这个屏障才能覆盖写入
    auto empty_barriers = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + (kNumStages + i) * static_cast<uint32_t>(sizeof(Barrier))));
    });

    // ========== 初始化 Barrier 和 Tensor Map ==========
    // 由一个专用的 warp (warp_idx == kNumMathThreads/32 + 1) 负责初始化
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // K-grouped GEMM 需要动态修改 Tensor Map 描述符
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            // 复制两份到 shared memory (双缓冲：当前组 + 下一组)
            *smem_tensor_map_a[0] = tensor_map_a_base;
            *smem_tensor_map_a[1] = tensor_map_a_base;
            *smem_tensor_map_b[0] = tensor_map_b_base;
            *smem_tensor_map_b[1] = tensor_map_b_base;
        }

        // 初始化 Cluster Transaction Barrier
        // full_barrier: 初始计数 1 (等待 1 个 TMA warp arrive)
        // empty_barrier: 初始计数 = 多播 CTA 数 × math warp 组数
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // 使 barrier 初始化在 async proxy 中可见
        // fence_barrier_init(): Hopper 专用指令，确保后续 wait 能看到初始值
        cutlass::arch::fence_barrier_init();
    }

    // ========== 全局同步 ==========
    // 确保所有 threads 都能看到初始化后的 barrier
    // kNumTMAMulticast > 1: 使用 cluster-wide 同步
    // 否则：使用 block-level __syncthreads()
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // ========== 流水线展开控制 ==========
    // K-grouped GEMM 不展开 (避免寄存器溢出)，其他情况完全展开
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);

    // ========== 寄存器重配置 ==========
    // Hopper 支持动态分配 registers 给不同的 warp groups
    // TMA warp: 少寄存器 (40/24), 更多 threads 可以同时活跃
    // Math warp: 多寄存器 (232/240), 存放 WGMMA 累加器
    constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
    constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

    // ========== Block Scheduler 初始化 ==========
    // Scheduler 负责将 blockIdx 映射到 (m_block, n_block) 坐标
    // 支持持久化调度：一个 block 可以处理多个 (m,n) 块
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, 128u>(shape_m, shape_n, shape_k, grouped_layout);

    // ========== 流水线阶段计算器 ==========
    // 返回 {stage_idx, phase}:
    // - stage_idx: 当前使用第几级流水线 (0 ~ kNumStages-1)
    // - phase: 奇偶相位 (0/1), 用于区分同一 stage 的不同轮次
    const auto& get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1}; // Pipeline stage and phase
    };
    uint32_t iter_idx = 0;  // 全局迭代计数器

    // ========== TMA Warp 路径：数据加载 ==========
    // warp_idx >= kNumMathThreads/32: 这些 warps 专职负责 TMA 数据传输
    if (warp_idx >= kNumMathThreads / 32) {
        // 释放寄存器给 TMA 操作 (只需要少量寄存器)
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // 只有一个 warp 实际执行 (warp_idx == kNumMathThreads/32)
        // cute::elect_one_sync(): 选举一个 leader thread 执行
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            const cute::TmaDescriptor* current_tensor_map_a = &tensor_map_a_base;
            const cute::TmaDescriptor* current_tensor_map_b = &tensor_map_b_base;
            uint32_t last_group_idx = kNumGroups;
            uint32_t prefetched_next_group_idx = kNumGroups;  // 追踪预取的下一组索引

            // ========== 持久化调度主循环 ==========
            // scheduler.get_next_block(): 获取下一个要处理的 (m_block, n_block)
            // 返回 false 时表示所有 blocks 处理完成
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // ========== TMA Multicast 配置 ==========
                // 判断当前 block 是否支持多播 (必须是偶数行/列，且未越界)
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                // 根据方向分配多播数量
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
                
                // 计算 K 维度的 block 数量和起始索引
                const uint32_t& num_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t& m_idx = m_block_idx * BLOCK_M;  // M 维度全局索引
                const uint32_t& n_idx = n_block_idx * BLOCK_N;  // N 维度全局索引

                // ========== K-Grouped GEMM 动态 Tensor Map 管理 ==========
                // K-grouped GEMM: 多个专家网络有不同的 K 维度
                // 需要动态修改 TMA descriptor 的 global address 和 stride
                if (kGemmType == GemmType::KGroupedContiguous and last_group_idx != scheduler.current_group_idx) {
                    const uint32_t& stage_idx = scheduler.current_num_valid_groups & 1;  // 双缓冲索引
                    const uint32_t& next_stage_idx = stage_idx ^ 1;  // 下一个缓冲索引
                    last_group_idx = scheduler.current_group_idx;

                    // [情况 1] 当前组与预取组不匹配：需要准备当前组的 tensor map
                    // 发生在 blocks 较少 (< num_SMs) 时，scheduler 跳过了某些组
                    if (scheduler.current_num_valid_groups > 0 &&
                        scheduler.current_group_idx != prefetched_next_group_idx) {
                        // 使用 scheduler.current_k_cumsum 正确追踪 K 偏移 (即使跳过组也能工作)
                        const uint64_t current_k_offset = scheduler.current_k_cumsum;
                        // 修改 shared memory 中的 tensor map
                        tensor_map_replace_global_addr_in_smem(smem_tensor_map_a[stage_idx],
                            gmem_a_ptr + current_k_offset * shape_m);
                        tensor_map_replace_global_addr_in_smem(smem_tensor_map_b[stage_idx],
                            gmem_b_ptr + current_k_offset * shape_n);
                        // 修改 inner dimension stride (不同组的 K 可能不同)
                        tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a[stage_idx],
                            scheduler.current_shape_k, scheduler.current_shape_k);
                        tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_b[stage_idx],
                            scheduler.current_shape_k, scheduler.current_shape_k);
                        // 同步到 global memory 缓冲区
                        *(gmem_tensor_map_a[stage_idx]) = *(smem_tensor_map_a[stage_idx]);
                        *(gmem_tensor_map_b[stage_idx]) = *(smem_tensor_map_b[stage_idx]);
                        // 注意：这里不调用 release，因为马上就要 acquire 当前组
                    }

                    // [情况 2] 预取下一组的 tensor map (隐藏 CPU 开销)
                    if (scheduler.next_group_idx < kNumGroups) {
                        // 计算下一组的 K 偏移
                        const uint64_t next_k_offset = static_cast<uint64_t>(scheduler.current_k_cumsum) + scheduler.current_shape_k;
                        // 准备下一组的 tensor map 到 shared memory
                        tensor_map_replace_global_addr_in_smem(smem_tensor_map_a[next_stage_idx], gmem_a_ptr + next_k_offset * shape_m);
                        tensor_map_replace_global_addr_in_smem(smem_tensor_map_b[next_stage_idx], gmem_b_ptr + next_k_offset * shape_n);
                        // 设置下一组的 inner dimension stride
                        tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a[next_stage_idx], scheduler.next_shape_k, scheduler.next_shape_k);
                        tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_b[next_stage_idx], scheduler.next_shape_k, scheduler.next_shape_k);
                        // 同步到 global memory
                        *(gmem_tensor_map_a[next_stage_idx]) = *(smem_tensor_map_a[next_stage_idx]);
                        *(gmem_tensor_map_b[next_stage_idx]) = *(smem_tensor_map_b[next_stage_idx]);
                        // 释放 CTA 锁：允许其他 CTAs 看到这个新的 tensor map
                        tensor_map_release_cta();
                        prefetched_next_group_idx = scheduler.next_group_idx;  // 记录预取了哪一组
                    } else {
                        prefetched_next_group_idx = kNumGroups;  // 没有更多组需要预取
                    }

                    // 获取当前组的 tensor map (acquire 锁)
                    if (scheduler.current_num_valid_groups > 0) {
                        tensor_map_acquire_cta(gmem_tensor_map_a[stage_idx]);
                        tensor_map_acquire_cta(gmem_tensor_map_b[stage_idx]);
                        current_tensor_map_a = gmem_tensor_map_a[stage_idx];
                        current_tensor_map_b = gmem_tensor_map_b[stage_idx];
                    }
                }

                // ========== K 维度软件流水线主循环 ==========
                // 遍历当前 block 的所有 K 分块
                #pragma unroll kNumPipelineUnrolls  // 根据配置展开或不展开
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                    // [步骤 1] 等待消费者 (Math warp) 释放缓冲区
                    // CUTE_TIE_DECL: 解包 tuple {stage_idx, phase}
                    // phase ^ 1: 等待上一轮的相位 (确保缓冲区已空)
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // [步骤 2] 发起 4 个 TMA 拷贝操作
                    // TMA 是异步硬件单元，发起后立既返回，不阻塞
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t& k_idx = k_block_idx * BLOCK_K;  // K 维度索引
                    const uint32_t& sf_k_idx = scheduler.current_sf_k_cumsum + k_block_idx;  // Scale factor 的 K 索引
                    
                    // TMA 拷贝 1: Scale Factor A [BLOCK_M, 1]
                    tma_copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], m_idx, sf_k_idx, num_tma_multicast_a);
                    // TMA 拷贝 2: Scale Factor B [BLOCK_N, 1]
                    tma_copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], n_idx, sf_k_idx, num_tma_multicast_b);
                    // TMA 拷贝 3: Matrix A [BLOCK_M, BLOCK_K] (带 swizzle)
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(current_tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
                    // TMA 拷贝 4: Matrix B [BLOCK_N, BLOCK_K] (带 swizzle)
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(current_tensor_map_b, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b);
                    
                    // [步骤 3] 通知 barrier：预期传输的字节数
                    // arrive_and_expect_tx(): Hopper 专用指令，精确跟踪 TMA 传输量
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                }
            }

            // ========== 清理分布式 Barrier ==========
            // 如果使用了 TMA multicast (>1 CTAs)，需要额外一轮 wait 来安全销毁 barrier
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                }
            }
        }
    } else {
        // ========== Math Warp 路径：WGMMA 计算 ==========
        // warp_idx < kNumMathThreads/32: 这些 warps 专职负责矩阵乘法计算
        // 分配更多寄存器用于存放 WGMMA 累加器
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // ========== Thread 坐标计算 ==========
        // __shfl_sync: 跨线程同步读取，鼓励 NVCC 使用统一寄存器
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);  // Math warp 组索引
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;  // 8x8 线程网格中的位置
        const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;  // 两行的起始索引 (间隔 8)

        // ========== Math Warp 持久化调度主循环 ==========
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // 检查 WGMMA 配置合法性
            DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2), "Invalid block sizes");
            const uint32_t& current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t& current_group_idx = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_group_idx : 0);
            const uint32_t& num_k_blocks = ceil_div(current_shape_k, BLOCK_K);
            
            // WGMMA 累加器数组
            float accum[WGMMA::kNumAccum];         // 原始 FP8×FP8 结果 (未反量化)
            float final_accum[WGMMA::kNumAccum] = {0};  // 反量化后的 FP32 累加结果
            float2 scales_b[WGMMA::kNumAccum / 4];      // B 的 scaling factors (成对加载)

            // ========== Empty Barrier 到达函子 ==========
            // 计算完成后通知 TMA warp: "缓冲区已空，可以覆盖"
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    // 无多播：lane_idx=0 的 thread 代表整个 warp arrive
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    // 有多播：需要指定目标 CTA
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // ========== K 维度流水线主循环 (Math 侧) ==========
            #pragma unroll kNumPipelineUnrolls
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                // [步骤 1] 等待 TMA 数据就绪
                CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);

                // [步骤 2] 从 shared memory 读取 Scaling Factors
                // 必须在 warpgroup_arrive 之前完成，避免被下一个 block 污染
                auto scale_a_0 = ld_shared(smem_sfa[stage_idx] + r_0);  // 第 r_0 行的 scale
                auto scale_a_1 = ld_shared(smem_sfa[stage_idx] + r_1);  // 第 r_1 行的 scale

                // [步骤 3] 读取 B 的 scaling factors (成对加载，提高效率)
                #pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    scales_b[i] = ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + i * 8 + col_idx * 2));

                // [步骤 4] 发起 WGMMA 矩阵乘法
                // warpgroup_fence_operand(): 确保累加器在 fence 外，防止重排序
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                
                // warpgroup_arrive(): 通知 warp 组开始 MMA 操作
                warpgroup_arrive();
                
                // WGMMA 循环：BLOCK_K / WGMMA::K 次迭代
                // WGMMA::K: 每条 WGMMA 指令处理的 K 维度元素数 (通常 32)
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    // 构造 shared memory 描述符
                    // math_wg_idx * WGMMA::M: 当前 warp 组负责的 M 起始位置
                    // k * WGMMA::K: 当前 K 分块的起始位置
                    auto desc_a = make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                    auto desc_b = make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                    // 发起 WGMMA 指令：accum += A @ B.T
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                
                // warpgroup_commit_batch(): 提交所有 WGMMA 指令到流水线
                warpgroup_commit_batch();
                
                // warpgroup_fence_operand(): 确保累加器不被后续操作重排序
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                
                // warpgroup_wait<0>(): 等待所有 WGMMA 操作完成 (0 表示等待所有)
                warpgroup_wait<0>();

                // [步骤 5] 通知 TMA warp: 缓冲区已空
                empty_barrier_arrive(stage_idx);

                // [步骤 6] FP8 反量化 + 累加
                // FP8 GEMM 的特殊操作：WGMMA 输出是 FP8×FP8 的中间结果
                // 需要乘以 scale factors 转换为 FP32 并累加
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const float &scale_b_0 = scales_b[i].x;  // B 的第 i 对 scale 的第一个
                    const float &scale_b_1 = scales_b[i].y;  // B 的第 i 对 scale 的第二个
                    
                    // 2x2 的反量化模式:
                    // final_accum[i*4+0] += scale_a_0 * scale_b_0 * accum[i*4+0]
                    // final_accum[i*4+1] += scale_a_0 * scale_b_1 * accum[i*4+1]
                    // final_accum[i*4+2] += scale_a_1 * scale_b_0 * accum[i*4+2]
                    // final_accum[i*4+3] += scale_a_1 * scale_b_1 * accum[i*4+3]
                    final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
                    final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
                    final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
                    final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
                }
            }

            // ========== Epilogue: 写回结果到 Global Memory ==========
            
            // [步骤 1] 刷新之前的 TMA store 操作
            if (warp_idx % 4 == 0 and cute::elect_one_sync())
                cute::tma_store_wait<0>();  // 等待所有 pending 的 stores 完成
            
            // [步骤 2] Named Barrier 同步 math warp 组
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // [步骤 3] 将 final_accum 写入 shared memory D 缓冲区
            // 使用 float2 成对存储，提高带宽利用率
            const auto& smem_d_0 = reinterpret_cast<float2*>(smem_d + r_0 * BLOCK_N + col_idx * 2);
            const auto& smem_d_1 = reinterpret_cast<float2*>(smem_d + r_1 * BLOCK_N + col_idx * 2);
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                st_shared(smem_d_0 + i * 4, {final_accum[i * 4 + 0], final_accum[i * 4 + 1]});
                st_shared(smem_d_1 + i * 4, {final_accum[i * 4 + 2], final_accum[i * 4 + 3]});
            }
            
            // TMA store fence: 确保 shared memory 写入对 TMA store 可见
            cute::tma_store_fence();
            // 再次同步 math warp 组
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // [步骤 4] 使用 TMA Store 写回 global memory
            // TMA Store 是异步的，由专用 warp 执行
            if (warp_idx % 4 == 0 and cute::elect_one_sync()) {
                // SM90_TMA_REDUCE_ADD_2D: 2D TMA store 原语
                // 参数：tensor map, shared memory 源，N/M 维度索引
                cute::SM90_TMA_REDUCE_ADD_2D::copy(
                    &tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N,
                    current_group_idx * shape_m + m_block_idx * BLOCK_M + r_0);
                // tma_store_arrive(): 通知 TMA store 单元开始传输
                cute::tma_store_arrive();
            }
            __syncwarp();  // 等待 warp 内所有 threads
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
