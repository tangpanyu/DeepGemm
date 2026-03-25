# DeepGEMM Kernel 变体详解

## 📋 总览：Kernel 家族分类

```
DeepGEMM Kernels
├── 基础 GEMM (正常矩阵乘法)
│   ├── fp8_gemm_nt/nn/tn/tt          # 标准 FP8 GEMM
│   ├── bf16_gemm_nt/nn/tn/tt         # BF16 GEMM
│   └── fp8_fp4_gemm_nt/nn/tn/tt      # FP8×FP4 混合精度 GEMM
│
├── Grouped GEMM (MoE 专用)
│   ├── M-Grouped (按 M 维度分组)
│   │   ├── m_grouped_*_contiguous    # 连续布局 (training/prefill)
│   │   └── m_grouped_*_masked        # 掩码布局 (decoding/CUDA Graph)
│   └── K-Grouped (按 K 维度分组)
│       └── k_grouped_*_contiguous    # K 方向分组 (MoE backward)
│
├── Attention 相关
│   ├── fp8_mqa_logits                # MQA logits 计算 (非分页)
│   └── fp8_paged_mqa_logits          # Paged MQA logits (分页)
│
├── 特殊场景
│   ├── tf32_hc_prenorm_gemm          # Hyperconnection 预归一化 GEMM
│   └── bmk_bnk_mn                    # BMK@BNK→MN 特殊布局
│
└── 辅助 Kernel
    ├── smxx_clean_logits             # 清理未填充的 logits
    └── smxx_layout                   # 布局转换工具
```

---

## 🎯 1. 基础 GEMM vs Grouped GEMM

### **基础 GEMM (Normal Dense GEMM)**

**作用**: 标准的矩阵乘法 `D = C + A @ B.T`

**使用场景**:
- 普通 Transformer 层的 FFN 网络
- Embedding 层
- 任何需要 `M×K @ N×K` 的地方

**API 示例**:
```python
import deep_gemm as dg

# 标准 NT 布局：[M, K] @ [N, K].T → [M, N]
dg.fp8_gemm_nt(a, b, d, c=None)

# 内存布局说明:
# a: [M, K] row-major (C 顺序)
# b: [N, K] col-major (F 顺序，即已经转置)
# d: [M, N] row-major
```

**特点**:
- 单一矩阵对，无分组
- 最简单、最常用
- 支持所有内存布局 (NT/TN/NN/TT)

---

### **Grouped GEMM (分组 GEMM)**

**核心思想**: 一次 kernel launch 处理**多个专家网络**的 GEMM

**为什么需要 Grouped GEMM?**

传统 MoE 实现:
```python
# ❌ 低效：多次 kernel launch
for expert in experts:
    output[i] = input[i] @ expert.weight
```

Grouped GEMM:
```python
# ✅ 高效：一次 kernel 处理所有专家
m_grouped_gemm(all_inputs, all_expert_weights, outputs, group_layout)
```

---

## 🔷 2. M-Grouped GEMM (按 M 维度分组)

### **应用场景**: MoE 模型中的专家共享相同形状 (N, K 固定),但每个专家处理的 token 数 (M) 不同

### **2.1 Contiguous Layout (连续布局)**

**文件**: [`sm90_fp8_gemm_1d1d.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh) (通过 `GemmType::MGroupedContiguous`)

**用途**: 
- Training forward pass
- Inference prefilling 阶段
- CPU 知道每个专家有多少 tokens

**数据布局**:
```
输入 A: [M_total, K]
├─ Expert 0: tokens [0:m₀] 
├─ Expert 1: tokens [m₀:m₀+m₁]
├─ Expert 2: tokens [m₀+m₁:m₀+m₁+m₂]
└─ ...

权重 B: [num_experts, N, K]
输出 D: [M_total, N]

grouped_layout: [m₀, m₁, m₂, ...]  # 每个专家的实际 token 数
```

**API**:
```python
dg.m_grouped_fp8_gemm_nt_contiguous(
    a,           # [M_total, K]
    b,           # [num_experts, N, K]
    d,           # [M_total, N]
    grouped_layout,  # int32 tensor, 记录每组的起始位置
    use_psum_layout=False
)
```

**关键约束**:
- `M_total` 必须对齐到 `get_mk_alignment_for_contiguous_layout()` (通常 128)
- 不足的部分用 0 padding

**内部实现**:
```cpp
// Scheduler 动态计算每个 block 属于哪个专家
if constexpr (kGemmType == GemmType::MGroupedContiguous) {
    const auto offset = __ldg(grouped_layout + m_block_idx * BLOCK_M);
    global_m = offset * shape_dim + local_m;
}
```

---

### **2.2 Masked Layout (掩码布局)**

**文件**: [`sm90_fp8_gemm_1d2d.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh) (通过 `GemmType::MGroupedMasked`)

**用途**:
- Inference decoding 阶段
- CUDA Graph 启用时
- CPU **不知道**每个专家的 token 数 (由 GPU 动态决定)

**数据布局**:
```
输入 A: [num_experts, max_tokens, K]
权重 B: [num_experts, N, K]
输出 D: [num_experts, max_tokens, N]

masked_m: [num_experts]  # 每个专家实际有效的 token 数
```

**API**:
```python
dg.m_grouped_fp8_gemm_nt_masked(
    a,              # [G, M_max, K]
    b,              # [G, N, K]
    d,              # [G, M_max, N]
    masked_m,       # 每个专家的有效 token 数
    expected_m      # 预期的最大 M
)
```

**示例**:
```python
# Decoding 阶段，CPU 不知道每个 expert 分到多少 tokens
num_experts = 8
max_tokens = 128
a = torch.randn(num_experts, max_tokens, K)
masked_m = torch.tensor([45, 67, 23, 89, 12, 56, 78, 34])  # 实际有效数

# Kernel 只会计算前 masked_m[i] 行，其余忽略
dg.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, max_tokens)
```

**内部实现**:
```cpp
// 根据 mask 跳过无效区域
if (m_block_idx >= masked_m[current_group]) 
    continue;  // 跳过这个 block
```

---

## 📐 3. K-Grouped GEMM (按 K 维度分组)

**文件**: [`sm90_bf16_gemm.hpp`](file:///home/tangpanyu/reps/DeepGEMM/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp#L244-L280) (通过 `GemmType::KGroupedContiguous`)

**用途**: **MoE 模型的 backward pass (权重梯度)**

**为什么需要 K-Grouped?**

Backward 计算公式:
```
∂L/∂W[g] = input[g].T @ ∂L/∂output[g]  # 每个专家的梯度
```

这里：
- `input[g]`: [M_g, K] - 不同专家有不同的 K
- `grad_output[g]`: [M_g, N]
- `grad_weight[g]`: [N, K] ← **要计算的目标**

**数据布局**:
```
输入 A: [K_total, M]  # 所有专家的 K 拼接
输入 B: [K_total, N]  # 所有专家的 K 拼接
输出 D: [num_experts, M, N]  # 每个专家的梯度

ks: [k₀, k₁, k₂, ...]  # 每个专家的 K 大小
```

**API**:
```python
dg.k_grouped_fp8_gemm_tn_contiguous(
    a,      # [sum(ks), M]
    b,      # [sum(ks), N]
    d,      # [num_experts, M, N]
    ks,     # List[int], 每个专家的 K 维度
    c=None  # optional accumulation
)
```

**示例**:
```python
# MoE backward
num_experts = 4
ks = [256, 512, 384, 448]  # 不同专家有不同的 K
M, N = 1024, 2048

a = torch.randn(sum(ks), M)  # gradients w.r.t input
b = torch.randn(sum(ks), N)  # gradients w.r.t output
d = torch.empty(num_experts, M, N)  # gradients w.r.t weights

dg.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks)
```

**内部实现**:
```cpp
// K-grouped scheduler 需要追踪 K 维度的累加
current_k_cumsum = sum(ks[0:current_group])
global_k = current_k_cumsum + local_k
```

---

## 🧠 4. MQA Logits Kernels (Attention 专用)

### **背景知识**: MQA (Multi-Query Attention)

```
Standard Multi-Head Attention:
Q: [seq_len, num_heads, head_dim]
K: [seq_len_kv, num_heads, head_dim]
V: [seq_len_kv, num_heads, head_dim]
Attention = softmax(Q @ K.T / sqrt(d)) @ V

MQA (共享 KV heads):
Q: [seq_len, num_heads, head_dim]
K: [seq_len_kv, 1, head_dim]  # 只有 1 个 head
V: [seq_len_kv, 1, head_dim]
```

---

### **4.1 Non-Paged MQA (非分页版本)**

**文件**: [`sm90_fp8_mqa_logits.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh)

**用途**: Prefilling 阶段，连续的 KV cache

**计算任务**:
```python
# 对于每个 token i:
kv_start = cu_seq_len_k_start[i]
kv_end = cu_seq_len_k_end[i]

for j in range(kv_start, kv_end):
    # Q[i] @ KV[j]
    scores = Q[i, :, :] @ (KV[j, :] * KV_scale[j])  # [num_heads]
    
    # ReLU + weight
    scores = relu(scores) * weights[i, :]
    
    # Sum over heads
    logits[i, j] = scores.sum()
```

**输入**:
- `q`: [seq_len, num_heads, head_dim] E4M3
- `kv`: [seq_len_kv, head_dim] E4M3
- `kv_scales`: [seq_len_kv] FP32
- `weights`: [seq_len, num_heads] FP32
- `cu_seq_len_k_start/end`: [seq_len] int32 (累积序列长度)

**输出**:
- `logits`: [seq_len, seq_len_kv] FP32

**API**:
```python
dg.fp8_mqa_logits(
    q, kv, kv_scales, weights,
    cu_seq_len_k_start, cu_seq_len_k_end,
    logits,
    clean_logits=True  # 是否清理未填充区域为 -inf
)
```

**典型场景**: DeepSeek V3.2 的 lightning indexer

---

### **4.2 Paged MQA (分页版本)**

**文件**: [`sm90_fp8_paged_mqa_logits.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh)

**用途**: Decoding 阶段，分页的 KV cache (vLLM 风格)

**与非分页的区别**:
- KV cache 不连续，分布在多个 blocks 中
- 需要 `block_table` 查找物理地址
- 支持 variable sequence lengths

**额外输入**:
- `block_table`: [batch_size, max_blocks] int32
- `context_lens`: [batch_size] int32
- `schedule_meta`: 调度元数据

**数据布局**:
```
Paged KV Cache:
kv_cache: [num_blocks, block_size, num_kv_heads, head_dim]
block_table: [batch_size, max_blocks_per_seq]

对于 seq i 的第 j 个 token:
    block_idx = block_table[i, j // block_size]
    offset = j % block_size
    kv = kv_cache[block_idx, offset]
```

**API**:
```python
# 先获取 metadata
meta = dg.get_paged_mqa_logits_metadata(block_table, context_lens, ...)

# 然后调用 paged kernel
dg.fp8_paged_mqa_logits(
    q, kv_cache, kv_scales, weights,
    context_lens, logits,
    block_table, meta,
    clean_logits=True
)
```

---

## 🔧 5. TF32 HC Prenorm GEMM (Hyperconnection)

**文件**: [`sm90_tf32_hc_prenorm_gemm.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh)

**用途**: Hyperconnection 架构的预归一化层

**什么是 Hyperconnection?**
- DeepSeek 模型的残差连接变体
- 在 LayerNorm 之前进行投影

**计算**:
```python
# Pre-norm + residual connection
x_norm = layernorm(x)
y = W @ x_norm  # GEMM part
output = x + y  # residual
```

**特殊性**:
- 输入是 TF32 (Tensor Float 32)
- 融合 pre-normalization
- 特殊的 epilogue (residual add)

---

## 📊 6. BMK@BNK→MN (特殊布局 GEMM)

**文件**: [`sm90_bmk_bnk_mn.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_bmk_bnk_mn.cuh)

**用途**: Batched GEMM with special memory layout

**数据布局**:
```
A: [B, M, K]  # Batch major
B: [B, N, K]  # Batch major
C: [M, N]     # Output (no batch dimension!)

计算：
for b in range(B):
    C += A[b] @ B[b].T
```

**与普通 GEMM 的区别**:
- 输入有 batch 维度
- 输出**没有**batch 维度 (累加到同一个矩阵)
- 类似 `torch.einsum('bmk,bnk->mn')`

**使用场景**:
- 多序列的 attention score 累加
- 某些特殊的 reduction 操作

---

## 🧹 7. Clean Logits Kernel (辅助功能)

**文件**: [`smxx_clean_logits.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/smxx_clean_logits.cuh)

**用途**: 将未填充的 logits 区域设为 `-inf`

**为什么需要?**
- MQA/Paged MQA 输出的 logits 可能有未定义区域
- Softmax 时需要这些区域为 `-inf`

**示例**:
```python
# MQA 输出: [seq_len, max_seq_len_kv]
# 实际有效：[seq_len, actual_seq_len_kv]
# 需要将 actual_seq_len_kv:max_seq_len_kv 设为 -inf

dg.smxx_clean_logits(
    logits,           # [seq_len, max_len]
    None,             # optional mask
    context_lens,     # [seq_len] 实际长度
    next_n,           # next token 位置
    batch_size,
    max_context_len
)
```

**内部实现**:
```cpp
// 简单的并行 fill
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= valid_range)
    logits[idx] = -INFINITY;
```

---

## 📐 8. Layout Kernel (布局转换)

**文件**: [`smxx_layout.cuh`](file:///home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/smxx_layout.cuh)

**用途**: 转换 scaling factors 的布局

**问题**: FP8 GEMM 需要特定格式的 scale factors
- SM90: FP32, per-128-channel
- SM100: UE8M0 packed (4 scales → 1 int32)

**API**:
```python
# 原始 scale: [M, K/128] FP32
sf_raw = torch.randn(M, K//128)

# 转换为 TMA 友好的布局
sf_transformed = dg.transform_sf_into_required_layout(
    sf_raw,
    target_layout='ue8m0_packed'  # or 'fp32_transposed'
)
```

---

## 🎨 完整使用场景对比表

| Kernel 类型 | 典型场景 | M 维度 | N 维度 | K 维度 | 分组方式 |
|------------|---------|--------|--------|--------|----------|
| **fp8_gemm_nt** | 普通 FFN | 固定 | 固定 | 固定 | ❌ 无 |
| **m_grouped_contiguous** | MoE training | 变化 | 固定 | 固定 | ✅ M 方向 |
| **m_grouped_masked** | MoE decoding | 变化 | 固定 | 固定 | ✅ M 方向 (mask) |
| **k_grouped_contiguous** | MoE backward | 固定 | 固定 | 变化 | ✅ K 方向 |
| **fp8_mqa_logits** | Attention prefill | seq_len | seq_len_kv | head_dim | ❌ 无 |
| **fp8_paged_mqa_logits** | Attention decode | batch | paged | head_dim | ✅ Paged |
| **tf32_hc_prenorm** | Hyperconnection | 固定 | 固定 | 固定 | ❌ 无 |
| **bmk_bnk_mn** | Batch reduction | batch×M | batch×N | K | ✅ Batch |

---

## 🔍 如何选择正确的 Kernel?

### **决策树**:

```
1. 是否是 MoE 场景？
   ├─ 否 → 使用 fp8_gemm_nt (基础版本)
   └─ 是 ↓

2. MoE 的哪个阶段？
   ├─ Forward (training/prefill) → m_grouped_*_contiguous
   ├─ Forward (decoding, CUDA Graph) → m_grouped_*_masked
   └─ Backward (weight gradient) → k_grouped_*_contiguous

3. 是否是 Attention?
   ├─ Prefill (连续 KV) → fp8_mqa_logits
   └─ Decode (分页 KV) → fp8_paged_mqa_logits

4. 特殊架构？
   ├─ Hyperconnection → tf32_hc_prenorm_gemm
   └─ Batch reduction → bmk_bnk_mn
```

---

## 💡 性能提示

### **Contiguous vs Masked**:
- Contiguous 更快 (无分支预测失败)
- 如果可能，优先使用 contiguous

### **Grouped GEMM 优化**:
```python
# ✅ 好的做法：对齐 M 维度
from deep_gemm import get_mk_alignment_for_contiguous_layout
alignment = get_mk_alignment_for_contiguous_layout()  # 通常 128
actual_ms = [align(m, alignment) for m in actual_ms]

# ❌ 坏的做法：不对齐导致 padding 开销
```

### **MQA Kernel**:
- 使用 `clean_logits=True` 避免额外的 kernel launch
- Paged 版本有额外的 page table lookup 开销 (~5-10%)

---

*最后更新：2026-03-25*
