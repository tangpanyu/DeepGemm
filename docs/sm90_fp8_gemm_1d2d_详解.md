# SM90 FP8 GEMM 1D2D Kernel 详解

## 📐 1D2D Tiling vs 1D1D Tiling 对比

### 1D1D Tiling (基础版本)
```
┌──────────────────────┐
│   Block(0,0)         │  ← 单个 block 处理整个 M×N
│                      │
│  整个 M×N 矩阵        │
│                      │
│  K 方向分块           │
└──────────────────────┘
```

### 1D2D Tiling (当前版本)
```
┌──────────┬──────────┬──────────┐
│Block(0,0)│Block(0,1)│Block(0,2)│  ← 多个 blocks 协作处理大矩阵
├──────────┼──────────┼──────────┤
│Block(1,0)│Block(1,1)│Block(1,2)│
├──────────┼──────────┼──────────┤
│Block(2,0)│Block(2,1)│Block(2,2)│
└──────────┴──────────┴──────────┘
每个 block 只处理 BLOCK_M × BLOCK_N 的一小块
```

---

## 🎯 核心差异

### 1. **Tiling 策略**

| 特性 | 1D1D | 1D2D |
|------|------|------|
| **M 维度** | 单 block 完整处理 | 分块，多 blocks 协作 |
| **N 维度** | 单 block 完整处理 | 分块，多 blocks 协作 |
| **K 维度** | 分块 (BLOCK_K=128) | 分块 (BLOCK_K=128) |
| **适用场景** | 中小矩阵 | 大矩阵 |
| **并行度** | 较低 | 更高 |

### 2. **Scheduler 调度器**

**1D1D**: 简单的线性调度
```cpp
while (scheduler.get_next_block(m_block_idx)) {
    // 只需要调度 M 维度
}
```

**1D2D**: 2D 网格调度
```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // 需要同时调度 M 和 N 维度
    // scheduler 会决定每个 block 负责哪一块
}
```

### 3. **Shared Memory 布局**

两者共享相同的 shared memory 布局:
```
+-----------------------+ <- 0x0
| D Accumulator Buffer  |   BLOCK_M * BLOCK_N * 2B (BF16)
+-----------------------+ <- SMEM_D_SIZE
| A Buffer (Stage 0)    |   BLOCK_M * BLOCK_K * 1B (FP8)
| B Buffer (Stage 0)    |   BLOCK_N * BLOCK_K * 1B (FP8)
| ...                   |   kNumStages 级流水线
+-----------------------+ <- + kNumStages * (SMEM_A + SMEM_B)
| Scale Factor A Buffer |   BLOCK_M * 4B (FP32)
| ...                   |   kNumStages 级
+-----------------------+ <- + kNumStages * ALIGNED_SMEM_SFA
| Scale Factor B Buffer |   shape_k_scales * (1 or 2) * 4B
+-----------------------+ <- 连续存储，非多级
| Barriers              |   2 * kNumStages 个屏障对象
+-----------------------+ <- 结束
```

---

## 🔍 关键代码解析

### 1. **模板参数**

```cpp
template <cute::UMMA::Major kMajorSFB,          // Scale Factor B 的布局
          uint32_t SHAPE_M, N, K,               // 全局矩阵维度
          uint32_t kNumGroups,                  // MoE 分组数
          uint32_t BLOCK_M, N, K,               // Block 级别分块
          uint32_t kSwizzleAMode, BMode, DMode, // Swizzle 模式
          uint32_t kNumStages,                  // 流水线级数
          uint32_t kNumTMAThreads,              // TMA warp 线程数 (128)
          uint32_t kNumMathThreads,             // Math warp 总线程数 (128/256)
          uint32_t kNumTMAMulticast,            // TMA 多播 CTA 数 (1-2)
          bool kIsTMAMulticastOnA,              // 多播方向
          uint32_t kNumSMs,                     // GPU SM 数量
          GemmType kGemmType,                   // GEMM 类型
          typename epilogue_type_t>             // Epilogue 类型
```

### 2. **Thread 角色分配**

```cpp
const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
const uint32_t lane_idx = get_lane_idx();

// TMA Warp (warp_idx >= kNumMathThreads/32)
if (warp_idx >= kNumMathThreads / 32) {
    // 负责数据加载 (Producer)
} else {
    // Math Warp (warp_idx < kNumMathThreads/32)
    // 负责 WGMMA 计算 (Consumer)
}
```

### 3. **TMA Warp 主循环 (Producer)**

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; 
         advance_pipeline(k_block_idx)) {
        
        // [步骤 1] 等待消费者释放缓冲区
        empty_barriers[stage_idx]->wait(phase ^ 1);
        
        // [步骤 2] 发起 TMA 拷贝 A 和 Scale Factor A
        tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, &full_barrier,
                 smem_a[stage_idx], k_idx, m_global_idx, ...);
        tma_copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                 smem_sfa[stage_idx], m_block_idx * BLOCK_M, sf_k_idx, ...);
        
        // [步骤 3] 发起 TMA 拷贝 B
        tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, &full_barrier,
                 smem_b[stage_idx], k_idx, n_global_idx, ...);
        
        // [步骤 4] 通知 barrier：预期传输的字节数
        full_barrier.arrive_and_expect_tx(total_bytes);
    }
}
```

### 4. **Math Warp 主循环 (Consumer)**

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // [前置] 提前加载 B Scales
    if (threadIdx.x >= 32) {
        // 分布式加载 scales 到 shared memory
        st_shared(smem_sfb + i, __ldg(local_sfb + offset));
    }
    
    if (scheduler.is_computation_valid(...)) {
        dispatch_num_former_iters<...>(..., [&](auto _) {
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; 
                 advance_pipeline(k_block_idx)) {
                
                // [步骤 1] 读取 B Scales
                float scale_b_0 = ld_shared(smem_sfb + k_block_idx);
                
                // [步骤 2] 等待 TMA 数据就绪
                full_barriers[stage_idx]->wait(phase);
                
                // [步骤 3] Wave 循环
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                    // [步骤 3a] 读取 A Scales
                    float scale_a_0 = ld_shared(smem_sfa[stage_idx] + r_0 + m_offset);
                    
                    // [步骤 3b] WGMMA 矩阵乘法
                    warpgroup_arrive();
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                        WGMMA::wgmma(a_desc, b_desc, accum, k);
                    }
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    
                    // [步骤 3c] 最后一个 wave 通知 empty barrier
                    if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                        empty_barrier_arrive();
                    
                    // [步骤 3d] FP8 反量化 + 累加
                    final_accum[...] += scale_a * scale_b * accum[...];
                }
            }
        });
    }
}
```

### 5. **Epilogue: STSM + TMA Store**

```cpp
// [步骤 1] 使用 STSM 指令写入 shared memory (带 swizzling)
SM90_U32x2_STSM_N<nv_bfloat162>::copy(
    __float22bfloat162_rn({accum[0], accum[1]}),
    __float22bfloat162_rn({accum[2], accum[3]}),
    smem_ptr  // 经过 XOR swizzle 的地址
);

// [步骤 2] TMA Store Fence
cute::tma_store_fence();

// [步骤 3] 发起 TMA stores 写回全局内存
if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
    cute::tma_store_arrive();
}
```

---

## 🚀 性能优化技术

### 1. **软件流水线 (Software Pipelining)**

```
时间 →
TMA:    [Load 0] [Wait] [Load 1] [Wait] [Load 2] [Wait]
        ┊        ┊      ┊        ┊      ┊        ┊
Math:          [Compute 0]  [Compute 1]  [Compute 2]
```

- **kNumStages = 4-8**: 隐藏 TMA 加载延迟
- **Barrier 同步**: full_barrier (数据就绪) + empty_barrier (缓冲区空)

### 2. **TMA Multicast (最多 2 CTAs)**

```
CTA 0 (TMA Warp): Load data ──┬──> Broadcast to CTA 0
                              └──> Broadcast to CTA 1 (if enabled)
```

- **kIsTMAMulticastOnA = true**: M 方向多播 (A 矩阵)
- **kIsTMAMulticastOnA = false**: N 方向多播 (B 矩阵)

### 3. **Shared Memory Swizzling**

```
XOR Swizzle: col ^= row % (swizzle_size / 16)

目的：避免 bank conflicts
效果：提高 shared memory 带宽利用率
```

### 4. **动态 Scaling Factor 加载**

```cpp
// 当 BLOCK_N 不能被 BLOCK_K 整除时
if constexpr (not kMustUseUniformedScaleB) {
    // 需要加载两行 scales
    scale_b_0 = ld_shared(smem_sfb + k_block_idx);
    scale_b_1 = ld_shared(smem_sfb + k_block_idx + shape_k_scales);
    
    // 根据位置选择正确的 scale
    const bool& predicate = i < num_former_iters;
    result += (predicate ? scale_0 : scale_1) * accum;
}
```

### 5. **寄存器重配置**

```cpp
// TMA Warp: 少寄存器 (40 个)
cutlass::arch::warpgroup_reg_dealloc<40>();

// Math Warp: 多寄存器 (232-248 个)
cutlass::arch::warpgroup_reg_alloc<232>();
```

---

## 📊 执行流程总览

```
Phase 1: 初始化
├─ Barrier 初始化
├─ TMA Descriptor 预取
└─ Shared memory 分区

Phase 2: 持久化调度循环
│
├─ TMA Warp (Producer)
│  ├─ 获取下一个 (m, n) block
│  ├─ K 循环:
│  │  ├─ 等待 empty barrier
│  │  ├─ TMA 加载 A + SF_A
│  │  ├─ TMA 加载 B
│  │  └─ Arrive full barrier
│  └─ 清理分布式 barriers
│
└─ Math Warp (Consumer)
   ├─ 获取下一个 (m, n) block
   ├─ 提前加载 B Scales
   ├─ K 循环:
   │  ├─ 读取 B Scales
   │  ├─ 等待 full barrier
   │  ├─ Wave 循环:
   │  │  ├─ 读取 A Scales
   │  │  ├─ WGMMA 计算
   │  │  ├─ 反量化累加
   │  │  └─ Arrive empty barrier (最后 wave)
   │  └─ End Wave
   ├─ STSM 写入 shared memory
   └─ TMA Store 写回全局内存

Phase 3: 结束
```

---

## 🔧 关键参数配置示例

### 配置 1: 中等规模矩阵
```python
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
kNumStages = 4
kNumMathThreads = 128
kSwizzleDMode = 128  # BF16, 128B swizzle
```

### 配置 2: 大规模矩阵
```python
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
kNumStages = 6
kNumMathThreads = 256
kSwizzleDMode = 128
```

---

## ❓ FAQ

### Q1: 为什么 1D2D 适合大矩阵？
**A**: 1D2D 在 M 和 N 维度都进行分块，允许多个 thread blocks 并行处理一个大矩阵的不同部分，而 1D1D 只有一个 block 处理整个 M×N。

### Q2: BLOCK_K 为什么固定为 128？
**A**: 这是 FP8 scaling factor 的粒度。每个 128-channel 需要一个 scaling factor，所以 BLOCK_K 固定为 128。

### Q3: Swizzle mode 如何选择？
**A**: 
- `kSwizzleAMode/BMode`: 通常为 32 或 64 (FP8 数据)
- `kSwizzleDMode`: 通常为 128 (BF16 数据)
- 目标：避免 shared memory bank conflicts

### Q4: 什么时候使用 TMA Multicast？
**A**: 当多个相邻 CTAs 访问相同的数据时使用。例如：
- M-grouped GEMM: 多个 experts 共享相同的 A 矩阵
- 可以节省 TMA 带宽，但增加同步开销

### Q5: `dispatch_num_former_iters` 的作用？
**A**: 运行时 dispatch 迭代次数，用于处理边界情况（当 BLOCK_N 不能被 BLOCK_K 整除时）。通过编译期展开循环优化性能。

---

## 📚 相关资源

- **1D1D 版本详解**: `sm90_fp8_gemm_1d1d_详解.md`
- **Kernel 变体详解**: `kernel_variants_explained.md`
- **CuTe 文档**: https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute
