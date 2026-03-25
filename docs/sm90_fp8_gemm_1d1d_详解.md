# SM90 FP8 GEMM 1D Tiling Kernel 详解

## 📚 核心概念

### 什么是 1D Tiling?

**1D Tiling** 是指在 **K 维度**上进行分块的策略:

```
传统 GEMM: C[M,N] = A[M,K] @ B[N,K].T

1D Tiling 方式:
┌─────────────┐      ┌─────────────┐
│   Block(0,0) │      │ 整个 M×N 由一个 thread block 处理 │
│  (BLOCK_M×BLOCK_N) │      │ K 方向分成多个小块           │
│             │      │ 每个小块 BLOCK_K=128          │
└─────────────┘      └─────────────┘
     ↓                      ↓
  [====K_0====][====K_1====]...[====K_n====]
       ←────────  流水线迭代  ────────→
```

**对比 2D Tiling**:
- **1D**: 一个 block 处理 `BLOCK_M × BLOCK_N` 的输出，只在 K 方向循环
  - 优点：简单，寄存器压力小
  - 缺点：大矩阵时 parallelism 不够
  
- **2D**: 多个 blocks 协作处理一个大矩阵，M 和 N 都分块
  - 优点：适合超大矩阵
  - 缺点：复杂，需要更多同步

---

## 🔧 Kernel 参数配置示例

典型的配置可能是:
```cpp
SHAPE_M = 0, SHAPE_N = 0, SHAPE_K = 0  // 运行时动态
BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 128
kSwizzleAMode = 32, kSwizzleBMode = 32  // shared memory swizzle
kNumStages = 4                          // 4 级流水线
kNumTMAThreads = 128                    // 4 warps
kNumMathThreads = 256                   // 8 warps (2 warp groups)
kNumTMAMulticast = 2                    // 2 CTAs 共享数据
kIsTMAMulticastOnA = true               // 在 M 方向多播
kNumSMs = 132                           // H100 有 132 个 SMs
kGemmType = GemmType::Normal
cd_dtype_t = float
```

---

## 🏗️ Kernel 执行流程

### 阶段 1: 初始化 (所有 threads)

```cpp
1. 编译期检查
   - TMA threads 必须是 128
   - BLOCK_K 必须是 128 (FP8 scaling 粒度)
   - C/D 必须是 float

2. Shared Memory 布局设计
   +-----------------------+ <- 0x0
   | Tensor Map Descriptors|   64B × 4 (仅 K-grouped 需要)
   +-----------------------+ <- SMEM_TENSOR_MAP_SIZE
   | D Accumulator Buffer  |   BLOCK_M × BLOCK_N × 4B
   +-----------------------+ 
   | A Buffer (Stage 0-N)  |   BLOCK_M × BLOCK_K × 1B × kNumStages
   | B Buffer (Stage 0-N)  |   BLOCK_N × BLOCK_K × 1B × kNumStages
   +-----------------------+
   | Scale Factor A Buffer |   BLOCK_M × 4B × kNumStages
   | Scale Factor B Buffer |   BLOCK_N × 4B × kNumStages
   +-----------------------+
   | Barriers              |   sizeof(Barrier) × 2 × kNumStages
   +-----------------------+
```

### 阶段 2: Thread 角色分配

```cpp
Thread 分工:
threadIdx.x ∈ [0, 127]           → TMA Warp (数据搬运)
threadIdx.x ∈ [128, 383]         → Math Warp Group 0 (WGMMA 计算)
threadIdx.x ∈ [384, 639]         → Math Warp Group 1 (WGMMA 计算)

warp_idx = threadIdx.x / 32
lane_idx = threadIdx.x % 32
```

### 阶段 3: Barrier 初始化

```cpp
full_barriers[i]: "数据就绪"屏障
  ├─ 初始值：1
  ├─ TMA arrive 后：0 → 触发 Math warp 开始计算
  └─ 用途：Producer → Consumer 同步

empty_barriers[i]: "缓冲区空"屏障
  ├─ 初始值：kNumTMAMulticast × kNumMathThreads/32
  ├─ Math arrive 后：递减 → 0 时触发 TMA 重新填充
  └─ 用途：Consumer → Producer 同步
```

---

## 🔄 软件流水线详解

### TMA Warp 主循环 (Producer)

```cpp
while (scheduler.get_next_block(m, n)) {
    for (k_block = 0; k_block < num_k_blocks; ++k_block) {
        // [步骤 1] 等待 empty barrier (phase ^ 1)
        empty_barriers[stage]->wait(phase ^ 1);
        
        // [步骤 2] 发起 4 个异步 TMA 拷贝
        tma_copy(sfa);  // Scale Factor A
        tma_copy(sfb);  // Scale Factor B  
        tma_copy(A);    // Matrix A (带 swizzle)
        tma_copy(B);    // Matrix B (带 swizzle)
        
        // [步骤 3] Full barrier arrive (预期传输字节数)
        full_barrier.arrive_and_expect_tx(total_bytes);
        
        // 进入下一级流水线
        stage = (stage + 1) % kNumStages;
        phase ^= (stage == 0);  // 完成一轮后翻转相位
    }
}
```

### Math Warp 主循环 (Consumer)

```cpp
while (scheduler.get_next_block(m, n)) {
    for (k_block = 0; k_block < num_k_blocks; ++k_block) {
        // [步骤 1] 等待 full barrier (phase)
        full_barriers[stage]->wait(phase);
        
        // [步骤 2] 读取 Scaling Factors (shared memory)
        scale_a = ld_shared(smem_sfa[stage]);
        scales_b = ld_shared(smem_sfb[stage]);
        
        // [步骤 3] WGMMA 矩阵乘法
        warpgroup_arrive();
        for (k = 0; k < BLOCK_K / WGMMA::K; ++k) {
            WGMMA::wgmma(desc_a, desc_b, accum, k);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        
        // [步骤 4] FP8 反量化 + 累加
        final_accum += scale_a * scales_b * accum;
        
        // [步骤 5] Empty barrier arrive
        empty_barriers[stage]->arrive();
        
        // 进入下一级流水线
        stage = (stage + 1) % kNumStages;
        phase ^= (stage == 0);
    }
    
    // Epilogue: 写回 global memory
    st_shared(smem_d, final_accum);
    tma_store(gmem_d, smem_d);
}
```

---

## ⚡ 关键优化技术

### 1. TMA Multicast (多播)

**问题**: 多个 CTAs 需要相同的数据块时，重复加载浪费带宽

**解决**: 硬件级多播
```cpp
// 2-CTA Cluster 配置
Cluster: [CTA_0, CTA_1]
           ↓
    只有 CTA_0 执行 TMA 加载
           ↓
    数据自动广播到 CTA_1
           ↓
    节省 ~50% L2 带宽
```

### 2. Swizzling (交织)

**问题**: Shared memory bank conflicts

**解决**: XOR-based swizzling
```cpp
// 无 swizzle: 顺序访问可能导致 bank conflict
addr = base + row * stride + col;

// 有 swizzle: 打散访问模式
addr = base + (row ^ (col / 32)) * stride + col;
```

### 3. 寄存器重配置

```cpp
// TMA warp: 少寄存器 → 高 occupancy
warpgroup_reg_dealloc<24>();

// Math warp: 多寄存器 → 存放累加器
warpgroup_reg_alloc<240>();
```

### 4. 持久化调度

```cpp
// 传统：1 block → 1 task
// 持久化：1 block → N tasks (while 循环)

while (scheduler.get_next_block(m, n)) {
    // 处理多个 (m,n) 块
}
```

---

## 🎯 WGMMA 指令解析

### WGMMA 是什么？

**WGMMA** = Warp Group Matrix Multiply Accumulate

- Hopper 架构专用指令
- 多个 warps 协作完成一个大矩阵乘法
- 比传统 MMA 更高的吞吐量

### 指令格式

```cpp
WGMMA::wgmma(desc_a, desc_b, accum, k_index);

// 实际执行的计算:
// accum[m,n] += ∑_k A[m,k] × B[n,k]
```

### 配置选择

```cpp
// 根据 BLOCK_N 选择合适的 WGMMA 变体
using WGMMA = typename FP8MMASelector<BLOCK_N>::type;

// 常见配置:
// BLOCK_N = 128 → WGMMA 128x128x32
// BLOCK_N = 256 → WGMMA 256x128x32
```

---

## 📊 性能考量

### Occupancy 分析

```
SM90 资源:
- Registers: 65536 per SM
- Shared Memory: 232 KB per SM
- Warps: 64 per SM

本 Kernel:
- TMA warp: 24 regs × 4 warps = 96 regs
- Math warps: 240 regs × 8 warps = 1920 regs
- Total: ~2016 regs ≈ 31% register file
- Occupancy: ~50% (受限于 shared memory)
```

###  Roofline 模型

```
H100 FP8 Tensor Core:
- Peak FP8: ~1979 TFLOPS
- HBM Bandwidth: ~3.35 TB/s

计算强度 (Arithmetic Intensity):
AI = FLOPs / Bytes = 2*M*N*K / (M*K + N*K + M*N)

对于大矩阵 (AI > 500):
- Compute-bound → 接近 peak TFLOPS

对于小矩阵 (AI < 100):
- Memory-bound → 受带宽限制
```

---

## 🔍 常见问题 FAQ

### Q1: 为什么 BLOCK_K 固定为 128?

**A**: FP8 的 scaling factor 是 per-128-channel 的
- 每 128 个 K 元素共享一个 scale
- 方便反量化计算

### Q2: Swizzle mode 如何选择？

**A**: 根据 shared memory 访问模式
- 32B: 适合小 block
- 64B/128B: 适合大 block，更好的 bank 分散

### Q3: 为什么需要两级 barrier (full/empty)?

**A**: 经典的 Producer-Consumer 模式
- full: Producer 通知 Consumer "数据好了"
- empty: Consumer 通知 Producer "可以覆盖了"

### Q4: TMA multicast 什么时候失效？

**A**: 边界情况
- M/N 不是 BLOCK_M/N 的整数倍
- 奇数 rows/columns
- Scheduler 会动态判断 `is_tma_multicast_valid()`

---

## 📖 进一步阅读

1. **NVIDIA Hopper Architecture Whitepaper**
   - 了解 WGMMA/TMA 硬件细节

2. **CUTLASS 3.x Documentation**
   - 更详细的 software pipelining 讲解

3. **CuTe DSL Tutorial**
   - 理解 tensor layout 和 copy 原语

4. **本项目的 sm100_fp8_gemm_1d1d.cuh**
   - 对比 Blackwell 架构的差异

---

*最后更新：2026-03-25*
