# Flash Attention 1 详解

## 1. 标准 Attention 怎么算？

在 Transformer 中，Self-Attention 的计算可以分为三步：

```
S = Q @ K^T          # (N, N) — 注意力分数矩阵
P = softmax(S)       # (N, N) — 注意力权重矩阵
O = P @ V            # (N, d) — 输出
```

其中 Q、K、V 的形状都是 (N, d)，N 是序列长度，d 是每个注意力头的维度。

关键观察：中间产物 S 和 P 都是 **N×N** 的矩阵。以 fp16 (2 bytes/元素) 为例：

| 序列长度 N | S 矩阵元素数 (N×N) | 单矩阵显存占用 (fp16) |
|-----------|-------------------|---------------------|
| 1,024     | 1,048,576         | 2 MB                |
| 4,096     | 16,777,216        | 32 MB               |
| 16,384    | 268,435,456       | 512 MB              |

N 翻 4 倍，显存占用翻 **16 倍**（平方关系）。而且 S 和 P 是两个 N×N 矩阵，实际开销还要翻倍。

但显存占用还不是最大的问题。真正的瓶颈是——**数据搬运**。

---

## 2. 真正的瓶颈：GPU 内存带宽

### GPU 的两级存储

GPU 有两级存储，速度差距巨大（以 NVIDIA A100 为例）：

```
┌─────────────────────────────────────┐
│           HBM（显存）                │
│   容量：80 GB                        │
│   带宽：2 TB/s                       │
│   Q, K, V, O 等张量都存放在这里       │
└──────────────┬──────────────────────┘
               │  ← 瓶颈在这里：数据搬运
┌──────────────▼──────────────────────┐
│         SRAM（片上共享内存）           │
│   容量：每个 SM 约 192 KB             │
│        全芯片 108 个 SM 共约 20 MB    │
│   带宽：约 19 TB/s（快 ~10 倍）       │
│   计算单元直接访问这里的数据           │
└─────────────────────────────────────┘
```

**类比：** HBM 像一个很大的仓库，SRAM 像工人手边的工作台。工作台很小但随手就能拿到东西；从仓库搬东西很慢，但仓库容量大。提高效率的关键是**减少去仓库搬运的次数**。

### 标准 Attention 的搬运开销

标准实现要在 HBM 和 SRAM 之间来回搬运 N×N 矩阵：

```
步骤 1: 从 HBM 读 Q, K  → 在 SRAM 中算出 S  → 把 S 写回 HBM     ← 写 N×N
步骤 2: 从 HBM 读 S     → 在 SRAM 中算出 P  → 把 P 写回 HBM     ← 读写 N×N
步骤 3: 从 HBM 读 P, V  → 在 SRAM 中算出 O  → 把 O 写回 HBM     ← 读 N×N
```

整个过程中，N×N 矩阵在 HBM 上被**写了 2 次、读了 2 次**，总搬运量为 O(N²)。对于长序列，这些搬运操作消耗的时间远超实际计算时间。

**Flash Attention 的目标：永远不把 N×N 矩阵写入 HBM。**

---

## 3. 核心障碍：softmax 需要全局信息

要实现"不存 N×N 矩阵"，最自然的想法是**分块计算**：把 Q、K、V 切成小块，每次只在 SRAM 中处理一小块。

矩阵乘法可以很容易地分块——两个大矩阵相乘可以拆成小块乘法之和。**但 softmax 不行。**

### 3.1 softmax 为什么不能分块？

softmax 的公式：

$$\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_{j=1}^{N} e^{s_j}}$$

分母 $\sum_{j=1}^{N} e^{s_j}$ 需要**遍历整行所有 N 个元素**才能算出来。这意味着：如果你只看到一行中的一部分数据，你无法算出这一部分的最终 softmax 值。

### 3.2 Safe Softmax：让问题变得更难

直接计算 $e^{s_i}$ 容易数值溢出（比如 $e^{1000}$ 就是 inf）。实际中必须用数值稳定版本：

$$\text{softmax}(s_i) = \frac{e^{s_i - m}}{\sum_{j=1}^{N} e^{s_j - m}}, \quad m = \max_{j}(s_j)$$

先减去最大值，保证指数的输入 $\le 0$ ，不会溢出。分子分母同时乘以 $e^{-m}$ ，结果不变。

但这意味着我们需要**两个**全局量：
1. **全局最大值** $m = \max(s_1, ..., s_N)$
2. **全局指数和** $l = \sum_{j=1}^{N} e^{s_j - m}$

标准实现需要 **3 趟遍历数据**：

```
第 1 趟：遍历所有元素，求 m = max(s)            ← 从 HBM 读一遍
第 2 趟：遍历所有元素，求 l = Σ exp(s_i - m)    ← 从 HBM 再读一遍
第 3 趟：遍历所有元素，求 o_i = exp(s_i-m)/l × v_i  ← 从 HBM 又读一遍
```

三趟遍历 = 三次从 HBM 读取整行数据。能减少吗？

---

## 4. Online Softmax：让 softmax 可以分块

### 4.1 核心思路

Online Softmax（Milakov & Giber, 2018）的核心发现是：**max 和 sum 可以在一趟遍历中同时递推维护**。

关键洞察：当我们看到新的数据后，最大值可能变大。但旧的 sum 是用旧的最大值算的——没关系，**乘以一个修正因子就能把旧结果调整到新基准**。

### 4.2 逐步推导

假设我们从左到右依次处理元素。处理完前 k 个元素后，我们维护两个量：

$$m^{(k)} = \max(s_1, ..., s_k)$$

$$l^{(k)} = \sum_{j=1}^{k} e^{s_j - m^{(k)}}$$

注意 $l^{(k)}$ 是以 $m^{(k)}$ 为基准算出的指数和。

现在第 k+1 个元素 $s_{k+1}$ 到来了：

**第一步，更新 max：**

$$m^{(k+1)} = \max(m^{(k)},\ s_{k+1})$$

**第二步，更新 sum（关键推导）：**

我们需要算出以新基准 $m^{(k+1)}$ 为标准的指数和：

$$l^{(k+1)} = \sum_{j=1}^{k+1} e^{s_j - m^{(k+1)}}$$

把前 k 项和第 k+1 项分开：

$$l^{(k+1)} = \underbrace{\sum_{j=1}^{k} e^{s_j - m^{(k+1)}}}_{\text{前 k 项，需要从旧基准修正}} + \underbrace{e^{s_{k+1} - m^{(k+1)}}}_{\text{新元素}}$$

前 k 项怎么从旧基准变为新基准？给指数加减一个常数：

$$e^{s_j - m^{(k+1)}} = e^{(s_j - m^{(k)}) + (m^{(k)} - m^{(k+1)})} = e^{s_j - m^{(k)}} \cdot e^{m^{(k)} - m^{(k+1)}}$$

$e^{m^{(k)} - m^{(k+1)}}$ 与 j 无关，可以提到求和外面：

$$\sum_{j=1}^{k} e^{s_j - m^{(k+1)}} = e^{m^{(k)} - m^{(k+1)}} \cdot \underbrace{\sum_{j=1}^{k} e^{s_j - m^{(k)}}}_{= l^{(k)}}$$

**最终递推公式：**

$$\boxed{l^{(k+1)} = l^{(k)} \cdot e^{m^{(k)} - m^{(k+1)}} + e^{s_{k+1} - m^{(k+1)}}}$$

理解这个公式的两种情况：
- 如果新元素 $s_{k+1} \le m^{(k)}$ ，则 $m^{(k+1)} = m^{(k)}$ ，修正因子 $e^{m^{(k)} - m^{(k+1)}} = e^0 = 1$ ，退化为简单累加
- 如果新元素更大， $m^{(k+1)} > m^{(k)}$ ，修正因子 $e^{m^{(k)} - m^{(k+1)}} < 1$ ，相当于把之前高估的指数值压缩下来

这样，**一趟遍历同时维护 max 和 sum**，把 3 趟变成 2 趟（还需要 1 趟算输出）。

### 4.3 更进一步：把输出 O 也融入递推

Online Softmax 把 3 趟变 2 趟。Flash Attention 更进一步：**把输出的计算也融入递推，变成 1 趟**。

我们最终需要的输出是：

$$o = \frac{\sum_{j=1}^{N} e^{s_j - m} \cdot v_j}{\sum_{j=1}^{N} e^{s_j - m}} = \frac{\tilde{o}^{(N)}}{l^{(N)}}$$

定义**未归一化的输出**：

$$\tilde{o}^{(k)} = \sum_{j=1}^{k} e^{s_j - m^{(k)}} \cdot v_j$$

用完全相同的技巧，推导 $\tilde{o}$ 的递推公式：

$$\tilde{o}^{(k+1)} = \sum_{j=1}^{k+1} e^{s_j - m^{(k+1)}} \cdot v_j$$

$$= e^{m^{(k)} - m^{(k+1)}} \cdot \underbrace{\sum_{j=1}^{k} e^{s_j - m^{(k)}} \cdot v_j}_{= \tilde{o}^{(k)}} + e^{s_{k+1} - m^{(k+1)}} \cdot v_{k+1}$$

$$\boxed{\tilde{o}^{(k+1)} = \tilde{o}^{(k)} \cdot e^{m^{(k)} - m^{(k+1)}} + e^{s_{k+1} - m^{(k+1)}} \cdot v_{k+1}}$$

**至此， $(m, l, \tilde{o})$ 三个量全部可以递推更新，只需 1 趟遍历。**

最后把结果归一化： $o = \tilde{o}^{(N)} / l^{(N)}$ 即为最终输出。

### 4.4 从单元素推广到块

前面为了推导方便，我们是"一个元素一个元素"地递推。但在实际 GPU 上，逐元素处理太慢了——GPU 擅长的是批量并行计算。所以 Flash Attention 把 K/V 切成若干块，**每次处理一整块**（Bc 行 K/V），而不是一个元素。

数学上完全一致：把单元素递推公式中的"第 k+1 个元素"换成"第 j 块（包含 Bc 个元素）"即可。

下面逐行讲解：

#### 第一部分：计算当前块的局部统计量

```python
Sij = Qi @ Kj.T                           # (Br, Bc)
```

Qi 是 Q 的第 i 块（Br 行），Kj 是 K 的第 j 块（Bc 行）。矩阵乘法得到一个 Br×Bc 的小矩阵——这就是注意力分数矩阵 S 的一个小块。因为 Br 和 Bc 都很小（比如 64 或 128），这个小矩阵完全放得进 SRAM。

```python
mij = rowmax(Sij)                         # (Br,)
```

对 Sij 的每一行取最大值。注意这只是**当前块内**的最大值，不是全局最大值。

```python
Pij = exp(Sij - mij)                      # (Br, Bc)
```

用当前块内的最大值做 safe softmax 的指数运算。减去 mij 保证数值不溢出。

```python
lij = rowsum(Pij)                         # (Br,)
```

对每行求和，得到当前块的指数和。同样只是局部的。

**到这里，我们有了当前块的三个局部量：mij（局部 max）、Pij（局部 exp 矩阵）、lij（局部指数和）。**

#### 第二部分：用局部量更新全局量

现在要把当前块的局部统计量合并到之前所有块的累积结果中。这正是 Online Softmax 递推公式的块级版本。

```python
m_new = max(m_old, mij)
```

全局最大值 = max(之前所有块的最大值, 当前块的最大值)。和单元素递推中 $m^{(k+1)} = \max(m^{(k)}, s_{k+1})$ 完全对应。

```python
correction_old = exp(m_old - m_new)        # 旧结果的修正因子
correction_new = exp(mij  - m_new)         # 新块的修正因子
```

这两个修正因子是理解整个算法的关键：

- **correction_old = exp(m_old - m_new)**：之前所有块的结果是用 m_old 作为基准算的。现在基准变成了 m_new（可能更大），需要把旧结果"压缩"一下。如果 m_new > m_old，这个值 < 1，起到压缩作用；如果 m_new = m_old（当前块没有更大的值），这个值 = 1，旧结果不变。

- **correction_new = exp(mij - m_new)**：当前块内部是用 mij 作为基准算的。同样需要统一到 m_new 基准下。如果 mij < m_new（当前块的 max 不是全局 max），这个值 < 1；如果 mij = m_new（当前块包含全局最大值），这个值 = 1。

**核心思想：两个修正因子把"不同基准下的指数值"统一到"同一个新基准"下，使得旧结果和新结果可以正确相加。**

```python
l_new = l_old * correction_old + lij * correction_new
```

统一基准后，直接相加就得到了全局指数和。展开来看：

- `l_old * correction_old`：之前所有块的指数和，从旧基准 m_old 修正到新基准 m_new
- `lij * correction_new`：当前块的指数和，从局部基准 mij 修正到新基准 m_new

对应单元素递推公式 $l^{(k+1)} = l^{(k)} \cdot e^{m^{(k)} - m^{(k+1)}} + e^{s_{k+1} - m^{(k+1)}}$ 。

#### 第三部分：更新输出 O

输出的递推需要注意一个细节：算法中维护的是**归一化后的** $O = \tilde{o}/l$ ，而不是未归一化的 $\tilde{o}$ 。这是因为最终我们需要的就是归一化结果，而且归一化后的值数值范围更稳定。

更新公式：

$$O_{new} = \frac{l_{old} \cdot e^{m_{old} - m_{new}} \cdot O_{old} + e^{m_{ij} - m_{new}} \cdot P_{ij} V_j}{l_{new}}$$

分子可以拆成两部分理解：

- $l_{old} \cdot e^{m_{old} - m_{new}} \cdot O_{old}$ ：将 $O_{old} = \tilde{o}_{old}/l_{old}$ 代入，得到 $e^{m_{old} - m_{new}} \cdot \tilde{o}_{old}$ ，即旧的未归一化输出修正到新基准
- $e^{m_{ij} - m_{new}} \cdot P_{ij} V_j$ ：当前块的贡献， $P_{ij} V_j$ 是局部基准下的加权求和，乘以 $e^{m_{ij} - m_{new}}$ 修正到新基准

两部分相加 = $\tilde{o}_{new}$ ，除以 $l_{new}$ = $O_{new}$ 。正确。

### 4.5 用数值例子验证

假设序列长度 N=6，head_dim=1（标量），block_size=3：

```
s = [2, 4, 1, 5, 3, 0]      # 注意力分数（一行）
v = [1, 2, 3, 4, 5, 6]      # value（一行）
```

**第一块 s=[2, 4, 1]，v=[1, 2, 3] 到来：**

```
m₁ = max(2, 4, 1) = 4

l₁ = e^(2-4) + e^(4-4) + e^(1-4)
   = e⁻² + e⁰ + e⁻³
   ≈ 0.1353 + 1.0000 + 0.0498
   = 1.1851

õ₁ = e^(2-4)×1 + e^(4-4)×2 + e^(1-4)×3
   = 0.1353×1 + 1.0000×2 + 0.0498×3
   ≈ 0.1353 + 2.0000 + 0.1494
   = 2.2847

O₁ = õ₁/l₁ = 2.2847/1.1851 ≈ 1.9278
```

**第二块 s=[5, 3, 0]，v=[4, 5, 6] 到来：**

```
m₂ = max(5, 3, 0) = 5
m_new = max(m₁, m₂) = max(4, 5) = 5

# 修正因子
correction_old = e^(4-5) = e⁻¹ ≈ 0.3679
correction_new = e^(5-5) = e⁰ = 1

# 更新指数和
l₂ = e^(5-5) + e^(3-5) + e^(0-5) = 1 + 0.1353 + 0.0067 = 1.1420
l_new = l₁ × correction_old + l₂ × correction_new
      = 1.1851 × 0.3679 + 1.1420 × 1
      = 0.4360 + 1.1420
      = 1.5780

# 更新未归一化输出
õ₂ = e^(5-5)×4 + e^(3-5)×5 + e^(0-5)×6 = 4.0000 + 0.6767 + 0.0404 = 4.7171
õ_new = õ₁ × correction_old + õ₂ × correction_new
      = 2.2847 × 0.3679 + 4.7171 × 1
      = 0.8406 + 4.7171
      = 5.5577

O_new = õ_new / l_new = 5.5577 / 1.5780 ≈ 3.5220
```

**验证：用标准 softmax 直接算**

```
m = max(2,4,1,5,3,0) = 5

exp 值: [e^(2-5), e^(4-5), e^(1-5), e^(5-5), e^(3-5), e^(0-5)]
      = [e⁻³,    e⁻¹,    e⁻⁴,    e⁰,     e⁻²,    e⁻⁵   ]
      ≈ [0.0498,  0.3679,  0.0183,  1.0000,  0.1353,  0.0067]

l = 0.0498 + 0.3679 + 0.0183 + 1.0000 + 0.1353 + 0.0067 = 1.5780 ✓

o = (0.0498×1 + 0.3679×2 + 0.0183×3 + 1.0000×4 + 0.1353×5 + 0.0067×6) / 1.5780
  = (0.0498 + 0.7358 + 0.0549 + 4.0000 + 0.6765 + 0.0402) / 1.5780
  = 5.5572 / 1.5780
  ≈ 3.5217 ✓（与分块结果一致，微小差异来自四舍五入）
```

### 4.6 为什么必须这样设计？

```
约束 1: softmax 需要全局 max 和 sum   →  看似必须存完整 N×N 矩阵
约束 2: SRAM 放不下 N×N 矩阵          →  必须分块处理
约束 3: 分块后看不到全局信息           →  需要增量更新

解法:   (m, l, õ) 三个量全部支持递推  →  只需 O(Br) 额外状态
        从而实现 1 趟遍历、永不物化 N×N 矩阵
```

**修正因子 $e^{m_{old} - m_{new}}$ 的本质：** "换基"操作。所有之前以 $m_{old}$ 为基准算出的指数值，乘上这个因子后就等价于以 $m_{new}$ 为基准重新计算——但不需要重新遍历之前的数据。这是 Online Softmax 能够工作的数学根基。

---

## 5. Flash Attention 完整算法

### 5.1 伪代码

```python
# 输入: Q, K, V ∈ R^(N×d)
# 输出: O ∈ R^(N×d)
# 分块: K/V 分成 Tc = ⌈N/Bc⌉ 块，Q/O 分成 Tr = ⌈N/Br⌉ 块

# 初始化
O = zeros(N, d)
l = zeros(N)          # 每行的指数和
m = full(N, -inf)     # 每行的最大值

for j in range(Tc):                    # 外循环：遍历 K/V 块
    Kj = K[j*Bc : (j+1)*Bc]           # 从 HBM 加载到 SRAM
    Vj = V[j*Bc : (j+1)*Bc]           # 从 HBM 加载到 SRAM

    for i in range(Tr):                # 内循环：遍历 Q 块
        Qi = Q[i*Br : (i+1)*Br]       # 加载第 i 块 query（输入，不会被修改）
        Oi = O[i*Br : (i+1)*Br]       # 加载第 i 块的累积输出（每轮迭代会被更新）
        mi = m[i*Br : (i+1)*Br]       # 第 i 块当前的行最大值
        li = l[i*Br : (i+1)*Br]       # 第 i 块当前的行指数和

        # ---- 以下全在 SRAM 中完成 ----

        # 1. 计算当前块的注意力分数
        Sij = Qi @ Kj.T               # (Br, Bc) — 小矩阵，放得下 SRAM

        # 2. 当前块的局部统计量
        mij = rowmax(Sij)             # (Br,)
        Pij = exp(Sij - mij)          # (Br, Bc)
        lij = rowsum(Pij)             # (Br,)

        # 3. Online Softmax 递推更新
        mi_new = max(mi, mij)
        li_new = li * exp(mi - mi_new) + lij * exp(mij - mi_new)

        # 4. 更新输出
        Oi = (li * exp(mi - mi_new) * Oi + exp(mij - mi_new) * Pij @ Vj) / li_new

        # 5. 写回 HBM
        O[i], m[i], l[i] = Oi, mi_new, li_new
```

### 5.2 循环顺序的选择与代价

Flash Attention 1 选择了 **K/V 外循环、Q 内循环**。这种安排下：

- **优点：** 每对 (Kj, Vj) 加载到 SRAM 后，被所有 Q 块复用，K/V 总共只从 HBM 读 1 遍
- **代价：** 每个 Q 块的 O/m/l 在每次外循环迭代中都要从 HBM 加载一次、写回一次，总共 Tc 次读写

如果反过来（Q 外循环，K/V 内循环），情况恰好对调：

- K/V 要被反复加载（Tr 次）
- 但 O/m/l 只需加载 1 次、写回 1 次——因为同一个 Q 块在整个内循环中一直留在 SRAM 里

仅从 Q/K/V 的读取量看，两种安排是对称的。但从 O/m/l 的读写看，**反过来的安排更优**——这正是 **Flash Attention 2** 的改进之一：将循环顺序调换为 Q 外循环、K/V 内循环，从而让 O/m/l 常驻 SRAM，减少了大量 HBM 读写。

---

## 6. IO 复杂度分析

| 方法 | HBM 读写总量 | 说明 |
|------|-------------|------|
| 标准 Attention | O(Nd + N²) | 读写 Q/K/V (Nd) + 读写 S/P (N²) |
| Flash Attention | O(N²d² / M) | M 为 SRAM 大小 |

Flash Attention 论文证明：当 $M \ge d^2$ 时（实际总是满足），Flash Attention 的 IO 复杂度严格小于标准方法。

**计算量不变，都是 O(N²d)，但数据搬运量大幅下降。**

---

## 7. Python 实现

```python
import torch
import math

def flash_attention_forward(Q, K, V, block_size=64):
    """
    Flash Attention 1 前向传播的纯 Python 模拟。
    Q, K, V: (batch, heads, seq_len, head_dim)
    """
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros(B, H, N, 1, device=Q.device)
    m = torch.full((B, H, N, 1), -math.inf, device=Q.device)

    scale = 1.0 / math.sqrt(d)

    Tc = math.ceil(N / block_size)
    Tr = math.ceil(N / block_size)

    for j in range(Tc):  # 外循环：K/V 块
        kv_s = j * block_size
        kv_e = min(kv_s + block_size, N)
        Kj = K[:, :, kv_s:kv_e, :]
        Vj = V[:, :, kv_s:kv_e, :]

        for i in range(Tr):  # 内循环：Q 块
            q_s = i * block_size
            q_e = min(q_s + block_size, N)
            Qi = Q[:, :, q_s:q_e, :]
            Oi = O[:, :, q_s:q_e, :]
            li = l[:, :, q_s:q_e, :]
            mi = m[:, :, q_s:q_e, :]

            # 1. 当前块的注意力分数
            Sij = (Qi @ Kj.transpose(-2, -1)) * scale  # (B, H, Br, Bc)

            # 2. 当前块的局部统计量
            mij = Sij.max(dim=-1, keepdim=True).values  # (B, H, Br, 1)
            Pij = torch.exp(Sij - mij)                  # (B, H, Br, Bc)
            lij = Pij.sum(dim=-1, keepdim=True)          # (B, H, Br, 1)

            # 3. Online Softmax 递推更新
            mi_new = torch.maximum(mi, mij)
            alpha = torch.exp(mi - mi_new)    # 旧结果的修正因子
            beta = torch.exp(mij - mi_new)    # 新块的修正因子
            li_new = li * alpha + lij * beta

            # 4. 更新输出
            Oi_new = (li * alpha * Oi + beta * (Pij @ Vj)) / li_new

            # 5. 写回
            O[:, :, q_s:q_e, :] = Oi_new
            l[:, :, q_s:q_e, :] = li_new
            m[:, :, q_s:q_e, :] = mi_new

    return O


# ---- 验证正确性 ----
if __name__ == "__main__":
    torch.manual_seed(42)
    B = 2    # batch size：同时处理 2 条序列
    H = 4    # heads：4 个注意力头（多头注意力）
    N = 256  # seq_len：每条序列 256 个 token
    d = 64   # head_dim：每个头的维度
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    # 标准 Attention
    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    O_standard = P @ V

    # Flash Attention
    O_flash = flash_attention_forward(Q, K, V, block_size=64)

    print(f"最大误差: {(O_standard - O_flash).abs().max().item():.2e}")
    # 预期输出: ~1e-6 量级
```

---

## 8. 反向传播（简述）

Flash Attention 的反向传播同样不物化 N×N 矩阵。关键技巧：

- **前向传播时只保存 Q, K, V, O, l, m**（不保存 S 和 P 这两个 N×N 矩阵）
- **反向传播时重新计算 S 和 P**（用保存的 Q, K, V 重新算），以计算换内存
- 这就是所谓的 **recomputation（重计算）** 策略

虽然重新计算增加了一些 FLOPS，但由于减少了大量 HBM 读写，整体速度反而更快。

---

## 9. 总结

| 维度 | 标准 Attention | Flash Attention |
|------|---------------|-----------------|
| 显存占用 | O(N²) — 存储 S 和 P | **O(N)** — 不存 N×N 矩阵 |
| HBM 读写量 | O(Nd + N²) | **O(N²d²/M)** — 大幅减少 |
| 计算量 (FLOPS) | O(N²d) | O(N²d) — **不变** |
| 核心技巧 | — | 分块 + Online Softmax + 重计算 |

**一句话总结：** Flash Attention 的计算量和标准 Attention 相同，但通过分块计算 + Online Softmax 递推，避免了在慢速显存中读写 N×N 中间矩阵，将 Attention 从"内存带宽受限"变为"计算受限"，从而大幅提速。
