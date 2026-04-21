# Transformer 注意力机制演进：MHA → MQA → GQA → MLA → DSA

> 作者视角：大学算法课老师；读者视角：正在学习 Transformer 结构的学生。
>
> 本文按「由简入深」的顺序讲解五种注意力变体，最后做对比与联系总结。

---

## 0. 前置：为什么要研究这些变体？

Transformer 的核心是 Self-Attention，它的计算公式是：

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

在**训练**时，我们一次性算完整的序列，瓶颈是计算量（FLOPs）；
而在**推理**时（特别是自回归生成，一次吐一个 token），为了避免重复计算历史 token 的 K、V，我们会把它们缓存下来，这就是著名的 **KV Cache**。

KV Cache 的显存占用 ≈ `2 × batch × seq_len × n_heads × d_head × n_layers × sizeof(dtype)`

- 当 `seq_len` 变大（长上下文）、`batch` 变大（高并发）时，KV Cache 会吃掉绝大部分显存；
- 注意这里 `2` 是 K 和 V 两个矩阵，`n_heads` 是瓶颈的关键因素。

**所以这四种变体的核心动机是一致的：在「效果」和「KV Cache / 推理开销」之间做权衡。**

---

## 1. MHA（Multi-Head Attention，多头注意力）

这是 2017 年《Attention Is All You Need》提出的原始形态，也是基线。

### 1.1 结构

设隐藏维度 `d_model`，头数 `h`，每个头的维度 `d_head = d_model / h`。

对每个 token 的输入 `x ∈ ℝ^{d_model}`，分别经过三组独立的线性变换得到：

- `Q_i = x W_i^Q`，`K_i = x W_i^K`，`V_i = x W_i^V`，其中 `i = 1, …, h`
- 每个头独立做 `softmax(QK^T/√d_k) V`，得到 `head_i`
- 拼接后再过输出投影：`MHA(x) = Concat(head_1, …, head_h) W^O`

#### 1.1.0 符号与维度表（先统一）

| 符号 | 含义 | 典型数值（GPT-2 small） |
|---|---|---|
| `x` | 一个 token 经 embedding + 前层处理后的向量 | `∈ ℝ^{d_model}` |
| `X` | 整条序列堆叠 | `∈ ℝ^{L × d_model}` |
| `d_model` | 模型隐藏维度 | 768 |
| `h` | 注意力头数 | 12 |
| `d_head` | 每个头的维度，通常 `= d_model / h` | 64 |
| `d_k` | Key 向量维度，在 MHA 里 = `d_head` | 64 |
| `L` | 当前序列长度 | 1024 |

记号说明：`W_i^Q` 中**上标 `Q` 是用途标签**（用来产生 Query），**下标 `i` 是头编号**，不是乘方、不是转置。

#### 1.1.1 第 1 行：生成 Q、K、V

$$Q_i = x\,W_i^Q,\quad K_i = x\,W_i^K,\quad V_i = x\,W_i^V, \quad i=1,\dots,h$$

**(a) 三组权重矩阵的角色**

`W_i^Q, W_i^K, W_i^V ∈ ℝ^{d_model × d_head}`，是**可学习参数**。每个头有独立一套，所以 MHA 共有 `3h` 个这样的矩阵。

**(b) 矩阵乘法与维度变化**

对整条序列：

```
X          (L × d_model)
W_i^Q      (d_model × d_head)
───────────────────────────────
Q_i        (L × d_head)    —— 该头里每个 token 的 Query
K_i, V_i   同理，都是 (L × d_head)
```

**(c) Q、K、V 的语义分工（检索类比）**

- **Query (Q)**：当前 token "想查什么" —— 搜索框输入的关键词；
- **Key (K)**：每个 token 暴露的"索引标签" —— 文档被检索时的关键字；
- **Value (V)**：每个 token 实际携带的"内容" —— 文档正文。

注意力的本质：**用 Q 去匹配所有 K，算出相关度，再按相关度加权取 V**。

**(d) 为什么要 h 套而不是 1 套？**

不同头可学习不同的"检索偏好"：语法关系、指代、局部 vs 全局等。单一套参数只能学到一种模式，表达力不足。

#### 1.1.1-补充：shape 细节与工程实现

初学者在这里常犯几个典型误解，专门列出来澄清。

**(e) 所有头共享同一个输入 X，不存在 `X_i`**

流程图里"三个箭头分叉"的含义是：**同一份 X 分别乘以三组不同的权重矩阵**，而不是"X 被切成三份"。

```
         X  (L × d_model)         ← 同一份输入，所有头都看这一份
         │
   ┌─────┼─────┐
   ▼     ▼     ▼
  ×W_i^Q ×W_i^K ×W_i^V           ← 每个头独立三组权重
   │     │     │
   ▼     ▼     ▼
  Q_i   K_i   V_i                ← 输出三个不同用途的张量
```

**(f) X 和 Q_i 的 shape 不一样**

| 张量 | Shape（符号） | Shape（GPT-2 small 数值） |
|---|---|---|
| `X` | `(L × d_model)` | `(1024 × 768)` |
| `Q_i` | `(L × d_head)` | `(1024 × 64)` |

序列长度 `L` 不变，但**特征维度从 `d_model` 被压到 `d_head`**（这里小了 h = 12 倍）。

原因：`W_i^Q` 本质是一个"**压缩投影**"——把 `d_model` 维压到 `d_head` 维，让每个头只在一个低维子空间里做注意力。

**(g) `W_i^Q` 的 shape 由矩阵乘法维度对齐反推**

```
 X         @    W_i^Q       =    Q_i
(L × d_model)   (d_model × d_head)   (L × d_head)
         └──── 这两个 d_model 必须对齐 ────┘     └─ 剩下 L 和 d_head ─┘
```

所以：

$$W_i^Q,\;W_i^K,\;W_i^V \in \mathbb{R}^{d_\text{model} \times d_\text{head}} = \mathbb{R}^{768 \times 64}$$

**(h) 工程实现：h 个小矩阵合并成 1 个大矩阵**

数学上我们写 `h` 个独立的 `W_i^Q`（每个 `768 × 64`）。但代码里**不会真的搞 12 个小矩阵** —— 那样启动 kernel 太多、GPU 利用率低。实际做法是**拼成一个大矩阵一次算完**：

```python
# 数学形式：12 个 (768 × 64) 的 W_i^Q
# 工程实现：1 个  (768 × 768)  的 W_Q

self.W_Q = nn.Linear(d_model, d_model)          # 权重 shape (768, 768)
Q_all = self.W_Q(X)                              # (L, 768)
Q = Q_all.view(L, h, d_head).transpose(0, 1)     # reshape → (h, L, d_head)
# 之后每个头就能独立做注意力；K、V 同理
```

reshape 那一步等价于"把大矩阵的列切成 12 份"，与 12 个独立 `W_i^Q` 数学上完全等价。

所以在 PyTorch / HuggingFace 源码里你常看到：

```python
self.q_proj = nn.Linear(d_model, d_model)  # 注意：是 d_model，不是 d_head！
```

**这是工程上的合并写法，底层数学仍是 h 份 `(d_model × d_head)`**。

**(i) 一张表锁死所有 shape**（`d_model=768, h=12, d_head=64, L=1024`）

| 量 | Shape（符号） | 数值 |
|---|---|---|
| `X`（输入） | `(L, d_model)` | `(1024, 768)` |
| `W_i^Q / W_i^K / W_i^V`（单头，数学视角） | `(d_model, d_head)` | `(768, 64)` |
| `W^Q / W^K / W^V`（合并，代码视角） | `(d_model, d_model)` | `(768, 768)` |
| `Q_i / K_i / V_i` | `(L, d_head)` | `(1024, 64)` |
| `head_i` | `(L, d_head)` | `(1024, 64)` |
| `Concat(head_1, …, head_h)` | `(L, h·d_head) = (L, d_model)` | `(1024, 768)` |
| `W^O` | `(d_model, d_model)` | `(768, 768)` |
| `MHA(X)` | `(L, d_model)` | `(1024, 768)` |

> **一句话记忆**：进 MHA 是 `d_model` 维，中间每个头压到 `d_head` 维干活，h 个头拼回 `d_model` 维再过 `W^O`，出来还是 `d_model` 维。

#### 1.1.2 第 2 行：每个头内部做 Scaled Dot-Product Attention

$$\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i$$

**(a) `Q_i K_i^T`：相关度打分**

```
Q_i        (L × d_head)
K_i^T      (d_head × L)
───────────────────────────
Q_i K_i^T  (L × L)
```

第 `(a, b)` 项 = "token a 的 Query 与 token b 的 Key 的点积相似度" = token a 对 token b 的**原始注意力分数**。

**(b) `÷ √d_k`：缩放（scaling）**

当 `d_k` 较大时，点积值量级会变大（近似服从 `N(0, d_k)`），softmax 会被"推到极端"——几乎全部概率集中在单个位置，梯度趋近 0，训练不动。

除以 `√d_k` 做方差归一化，让分数量级与 `d_k` 无关，softmax 才有合理的"温度"。这是原论文专门强调的关键细节。

**(c) `softmax(...)`：归一化成注意力权重**

按行做 softmax，得到权重矩阵 `A ∈ ℝ^{L × L}`，每行求和为 1。在自回归解码器中，这里还会叠加**因果 mask**（未来位置填 `-∞`，softmax 后为 0），防止偷看未来。

**(d) `... V_i`：按权重加权聚合内容**

```
A          (L × L)
V_i        (L × d_head)
───────────────────────────
head_i     (L × d_head)
```

第 a 行结果 = "按 token a 的关注分布，把所有 token 的 V 加权平均"——这就是 token a 在**该头视角下**吸收到的信息。

#### 1.1.3 第 3 行：多头拼接 + 输出投影

$$\text{MHA}(x) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O$$

**(a) `Concat(head_1, …, head_h)`**

每个 `head_i` 是 `(L × d_head)`，按最后一维拼接：

```
Concat 结果   (L × h·d_head) = (L × d_model)
```

这也解释了为何通常 `d_head = d_model / h`——拼完正好回到 `d_model`，方便残差连接。

**(b) `W^O ∈ ℝ^{d_model × d_model}`：输出投影**

作用有二：

1. **混合各头信息**：拼接只是物理堆叠，`W^O` 的每一行是对所有头输出的一次加权组合，让它们真正"互相商量"；
2. **保持维度一致**：输入输出都是 `d_model`，方便接残差（`x + MHA(x)`）和 LayerNorm。

**(c) 最终形状** `MHA(X) ∈ ℝ^{L × d_model}`：每个 token 维度不变，但已融入从其它 token 收集到的信息。

#### 1.1.4 全流程示意图

以 `d_model=768, h=12, d_head=64, L=1024` 为例：

```
          输入 X  (1024 × 768)
             │
   ┌─────────┼─────────┐   做 h=12 次（每个头一套权重）
   ▼         ▼         ▼
 Q_i       K_i       V_i               (1024 × 64)
   │         │         │
   └──── Q_i K_i^T / √64 ─────→  (1024 × 1024)  打分
                                      │
                                   softmax    ← 可选 causal mask
                                      │
                                     × V_i
                                      │
                                 head_i  (1024 × 64)
                                      │
                              （h 个 head_i 拼接）
                                      │
                              Concat → (1024 × 768)
                                      │
                                   × W^O  (768 × 768)
                                      │
                              MHA(X)  (1024 × 768)
```

#### 1.1.5 三行公式一句话总结

| 公式行 | 做什么 | 类比 |
|---|---|---|
| `Q_i = x W_i^Q` 等 | 把 token 投影到 h 组不同的 (Q, K, V) 空间 | 准备 h 份不同视角的"关键词、索引、内容" |
| `softmax(QK^T/√d_k) V` | 每个头内部：打分 → 归一 → 加权取 V | 一次完整的"按关键词检索并汇总资料" |
| `Concat(...) W^O` | 拼接 h 个头，再混合一次 | 把 h 位"分析师"的报告汇总成最终结论 |

> 理解了这三行，MQA / GQA / MLA 的改动都只是在"第 1 行 K、V 那部分如何节省"上做文章——核心骨架完全一样。

### 1.2 关键特征

- **每个头都有自己独立的 Q、K、V 投影权重**；
- KV Cache 大小正比于 `h`（头数）；
- 表达能力最强，不同头可以学到不同的注意力模式（如语法关系、指代、局部 vs 全局等）。

### 1.3 示意图

```
Head 1: Q1  K1  V1
Head 2: Q2  K2  V2     ← 每个头都有独立的 K、V
...
Head h: Qh  Kh  Vh
```

### 1.4 代价

KV Cache 随头数线性增长。在 LLaMA 这类 32~80 个头的模型里，KV 成为了推理显存的大头。

---

## 2. MQA（Multi-Query Attention，多查询注意力）

Noam Shazeer 在 2019 年《Fast Transformer Decoding》里提出，被 PaLM、Falcon 等早期采用。

### 2.1 核心思想

> **保留多个 Q 头，但让所有头共享同一组 K、V。**

### 2.2 结构

- `Q_i = x W_i^Q`，i = 1, …, h（还是 h 套）
- `K = x W^K`，`V = x W^V`（**只有 1 套！**）
- 每个头用自己的 `Q_i` 去和**同一个** K、V 做注意力。

### 2.3 示意图

```
Head 1: Q1  ┐
Head 2: Q2  ├──→  共享的  K, V
...         │
Head h: Qh  ┘
```

### 2.4 收益与代价

- **KV Cache 直接缩小到 1/h**，推理显存和带宽压力骤减；
- 推理吞吐大幅提升（尤其在长上下文 / 大 batch 下）；
- **代价：效果会有一定退化**，因为原本不同头能学到不同的 K、V 空间，现在被强制压到同一个空间。

---

## 3. GQA（Grouped-Query Attention，分组查询注意力）

Google 在 2023 年的《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》提出，被 LLaMA 2 / 3、Mixtral 等主流模型采用。

### 3.1 核心思想

> **MHA 和 MQA 的折中：把 h 个 Q 头分成 g 组，每组共享一套 K、V。**

- 当 `g = h` 时，就是 MHA；
- 当 `g = 1` 时，就是 MQA；
- 常见实践中 `g = 8`（LLaMA-2 70B）或 `g = h/4` 等。

### 3.2 示意图

```
Group 1: Q1, Q2, Q3, Q4  ──→  K1, V1
Group 2: Q5, Q6, Q7, Q8  ──→  K2, V2
...
Group g: ...             ──→  Kg, Vg
```

### 3.3 特点

- **KV Cache 大小 ≈ MHA 的 g/h**（例如 8/64 = 1/8）；
- 效果非常接近 MHA，远好于 MQA；
- 可以**从现成的 MHA checkpoint 通过少量继续训练（uptraining）得到**——把同组的 K、V 权重做 mean pooling 即可初始化。

这也是 GQA 成为当前主流的关键原因：**它和已有 MHA 模型兼容，性价比最高**。

---

## 4. MLA（Multi-head Latent Attention，多头潜在注意力）

DeepSeek 在 DeepSeek-V2（2024）提出，DeepSeek-V3 / R1 延续使用。这是目前在推理效率上最激进、效果又没掉的设计。

### 4.1 动机

GQA 已经把 KV Cache 缩小了，但它是「砍头」式的压缩——直接减少 K、V 的数量。
DeepSeek 的思路不同：**能不能保留多头的表达能力，同时把 KV 在一个更小的潜在空间里压缩存储？**

### 4.2 核心思想：低秩联合压缩（Low-Rank Joint Compression）

> **把 K 和 V 联合投影到一个更小的潜在向量 `c_{KV}` 上缓存；使用时再从 `c_{KV}` 解压出多头的 K、V——而且"解压"这一步在推理时可以通过"权重吸收"直接绕过。**

下面用 8 个递进步骤，配具体 shape 和数值，把这一段讲透。

#### 4.2.1 第 1 步：先回忆 MHA 的 KV Cache 在存什么

MHA 里每个 token 每层每个头需要缓存一对 `(K_i, V_i)`，每个都是 `d_head` 维。

**每 token 每层要缓存的参数量 = `2 × h × d_head`**

以 DeepSeek-V2（`h=128, d_head=128`）为例：

```
每 token 每层 KV Cache = 2 × 128 × 128 = 32,768 个浮点数
```

fp16 下是 **64 KB / token / layer**，60 层 × 4K 上下文就是上百 GB——这就是推理瓶颈。

#### 4.2.2 第 2 步："低秩"到底是什么意思？

先丢开 attention，单看一个普通的大线性层：

$$y = x\,W,\quad x \in \mathbb{R}^{5120},\; W \in \mathbb{R}^{5120 \times 16384},\; y \in \mathbb{R}^{16384}$$

`W` 有 `5120 × 16384 ≈ 8400 万`参数。如果我们认为它的"有效信息量"没这么大，可以**分解成两个瘦长矩阵的乘积**：

$$W \approx W^D \cdot W^U,\quad W^D \in \mathbb{R}^{5120 \times 512},\; W^U \in \mathbb{R}^{512 \times 16384}$$

参数量：`5120×512 + 512×16384 ≈ 1100 万`，**压缩 8 倍**。

这就是 **low-rank decomposition（低秩分解）**。`512` 是潜在维度 `d_c`，`D` 指 down（下投影），`U` 指 up（上投影）。

**关键观察**：先算 `c = x W^D`（512 维的小向量），再算 `y = c W^U`（回到 16384 维）。**中间量 `c` 小，但最终 `y` 还是高维，表达力"几乎"不损失**（只要秩 512 足够）。

> **这就是 MLA 的底座：把 K、V 的生成拆成「低维压缩 → 高维解压」两步，把缓存对象从高维 K/V 换成低维的中间量 c。**

#### 4.2.3 第 3 步：具体 shape 走一遍 MLA 的 K、V 生成

设 `d_model=5120, h=128, d_head=128, d_c=512`（注意 `d_c=512 ≪ h·d_head=16384`）。

**(a) 下投影（compression），结果就是要缓存的对象：**

$$c_{KV} = x\,W^{DKV}$$

| 张量 | Shape |
|---|---|
| `x` | `(1, 5120)` |
| `W^{DKV}` | `(5120, 512)` |
| **`c_{KV}`** | **`(1, 512)`**  ← **只缓存这个** |

**(b) 上投影（decompression），推理用时再算，不缓存：**

$$K = c_{KV}\,W^{UK},\quad V = c_{KV}\,W^{UV}$$

| 张量 | Shape |
|---|---|
| `W^{UK}` | `(512, 16384)` → reshape `(512, h, d_head)` |
| `K` | `(1, 128, 128)` —— 128 个头，每个头 128 维 |
| `W^{UV}` | `(512, 16384)` |
| `V` | `(1, 128, 128)` |

有了 `K, V` 后再按普通 MHA 做多头注意力。

**(c) Q 侧也做一次同样的低秩压缩**（这一步跟 KV Cache 无关，只是为了省训练显存）：

$$c_Q = x\,W^{DQ} \in \mathbb{R}^{d_c'},\quad Q = c_Q\,W^{UQ}$$

#### 4.2.4 第 4 步：KV Cache 到底省了多少？

| 方法 | 每 token 每层缓存什么 | 数值（DeepSeek-V2 config） |
|---|---|---|
| MHA | `K, V`，共 `2 × h × d_head` | 32,768 |
| MQA | `K, V`，共 `2 × d_head` | 256 |
| GQA (g=8) | `K, V`，共 `2 × g × d_head` | 2,048 |
| **MLA** | **只缓存 `c_{KV}` + RoPE 分量** | **≈ 512 + 64 ≈ 576** |

MLA 比 MHA 小 **~57 倍**，甚至比 MQA 还有竞争力——关键是它**效果接近甚至好过 MHA**，而 MQA 会掉点。

### 4.3 为什么"几乎不掉点"？——权重吸收（Weight Absorption）

#### 4.3.1 第 5 步：引出"权重吸收"

上面你一定会产生这个疑问：

> 虽然缓存的是 512 维的 `c_{KV}`，**但每步推理还得把它上投影回 16384 维的 K 和 V，这不是照样一大坨矩阵乘？显存省了，FLOPs 没省啊？**

**MLA 最精妙的地方就在这里——实际上根本不需要真的解压出 K 和 V。** 这就叫 **权重吸收（weight absorption / matrix absorption）**。

#### 4.3.2 第 6 步：Q·K 侧的吸收推导

注意力分数要算 `Q K^T`。把 Q、K 都代换成原始变量：

- `Q = c_Q · W^{UQ}`
- `K = c_{KV} · W^{UK}`

代入：

$$Q\,K^\top = (c_Q\,W^{UQ})\,(c_{KV}\,W^{UK})^\top = c_Q\,W^{UQ}\,(W^{UK})^\top\,c_{KV}^\top$$

**关键观察**：中间那一坨 `W^{UQ} · (W^{UK})^T` **全是固定权重，不依赖输入**。推理开始前可以**一次性预先算出来**，记作：

$$\widetilde{W}_i = W_i^{UQ}\,(W_i^{UK})^\top \in \mathbb{R}^{d_c' \times d_c}\quad(\text{每个头 }i\text{ 一份})$$

shape 是 `(1536, 512)`—— **一个小矩阵**。

于是每次推理时，注意力分数直接算：

$$\text{score}_i = c_Q\,\widetilde{W}_i\,c_{KV}^\top$$

**注意这里的奇迹**：
- 全程从未真正算出 `K`（16384 维的大张量）；
- 只用了压缩态 `c_Q`（1536 维）和 `c_{KV}`（512 维）；
- 中间是一个预先吸收好的小矩阵 `W̃_i`。

**注意力运算全程在"压缩空间里"完成，既省显存又省算力。**

#### 4.3.3 第 7 步：V 侧吸收到输出投影 W^O 里

输出端同样有这个技巧。回忆 MHA 的输出公式：

$$\text{MHA} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

每个头：`head_i = A_i · V_i`，其中 `A_i` 是 softmax 后的注意力权重矩阵，`V_i = c_{KV} · W_i^{UV}`。

把 `W^O` 按头切片成 `W_i^O`（每个头对应 `d_head` 列），那么：

$$\text{MHA} = \sum_i \text{head}_i\,W_i^O = \sum_i A_i\,c_{KV}\,W_i^{UV}\,W_i^O$$

合并最后两个权重：

$$\widehat{W}_i^O = W_i^{UV}\,W_i^O \quad(\text{固定权重，可预算})$$

于是：

$$\text{MHA} = \sum_i A_i\,c_{KV}\,\widehat{W}_i^O$$

**同样：V 从未被真正还原出来。** 权重 `W^{UV}` 被"吸收"进了 `W^O`。

#### 4.3.4 第 8 步：推理时的完整流水对比

**MHA 推理一步：**

```
每层：
  ① 从 cache 读出 K, V         ← 读 32,768 个数 / token
  ② Q · K^T → softmax → · V    ← 大矩阵乘法
  ③ Concat · W^O
```

**MLA 推理一步（权重吸收后）：**

```
一次性预处理（推理启动时做一次）：
  W̃_i   = W_i^{UQ} · (W_i^{UK})^T    （每个头一个小矩阵）
  Ŵ_i^O = W_i^{UV} · W_i^O

每层每步：
  ① 从 cache 读出 c_{KV}         ← 只读 512 个数 / token   ← 显存带宽骤降
  ② 算 c_Q = x · W^{DQ}
  ③ score_i = c_Q · W̃_i · c_{KV}^T   ← 全在 512 维空间做
  ④ A_i = softmax(score_i / √d)
  ⑤ 输出 = Σ_i  A_i · c_{KV} · Ŵ_i^O
  ⑥ 把 c_{KV}^{new} = x · W^{DKV} 存回 cache
```

#### 4.3.5 一句话总结

> **MLA = 把 K、V 压成一个共享的低维潜在向量 `c_{KV}` 缓存；推理时不真的解压，而是把「上投影权重」数学上吸收进 Q 和 O 的投影里，让整个注意力运算发生在低维压缩空间内。**

显存小了几十倍、FLOPs 几乎不增加、效果还不掉——这就是 DeepSeek-V2 能"又便宜又强"的核心秘密。

### 4.4 背景补充：RoPE 是什么？

在讲 MLA 的 RoPE 兼容问题之前，先把 RoPE 本身解释清楚——前文已经用过几次这个名字，但没正式介绍。

#### 4.4.1 位置编码要解决什么问题？

注意力的公式 `softmax(QK^T/√d_k) V` 本身是**对 token 顺序不敏感的**：如果你把输入序列的顺序打乱，每个 token 得到的注意力输出并不会变化（只是位置换了）。这对语言模型是致命的——"猫追狗"和"狗追猫"语义完全不同。

所以必须给模型**注入"位置信息"**，这就叫 **position encoding（位置编码）**。

常见方案演化：

| 方案 | 做法 | 问题 |
|---|---|---|
| Sinusoidal（原版 Transformer） | 在 embedding 上直接**加**一个位置向量 | 外推到更长序列效果差 |
| Learned absolute | 每个位置学一个向量 | 长度固定，不能超出训练长度 |
| **RoPE**（RoFormer, 2021） | 对 Q、K 做**位置相关的旋转** | 外推能力较好、可解释性强 |
| ALiBi | 直接在注意力分数上加 `-m·|i-j|` 惩罚 | 简单但表达力略受限 |

LLaMA、Qwen、DeepSeek 等主流开源 LLM 现在**几乎都用 RoPE**。

#### 4.4.2 RoPE 的核心思想：用"旋转"表达位置

> **把 Q、K 向量在二维平面上按"位置相关的角度"旋转，让注意力分数 `Q·K^T` 天然只依赖于两 token 的相对位置。**

下面把这句话里的"凑对 → 旋转 → 不同频率"逐步拆开。

##### (a) 为什么要"凑对"？——旋转是二维操作

"旋转"在数学上**必须至少两个维度才有意义**：

- 1 维上只有"正负反向"；
- 2 维平面上才能说"转 30 度"、"转 90 度"；
- 更高维里旋转也总是围绕某个 2 维平面进行。

RoPE 要用旋转来编码位置，就只能**把高维向量切成若干个 2 维子空间，每个子空间里独立地转**。

- `d_head` 维向量 → 切成 `d_head/2` 个二维平面 → 每个平面里做 2D 旋转；
- 所以 **`d_head` 必须是偶数**（这也是为什么实际模型里 `d_head` 永远是 64、128、256 这种偶数）。

##### (b) "凑对"具体指什么？

以 `d_head = 8` 为例，某个头的 Query 向量是：

```
q = [q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7]
```

"两两凑对"就是把 8 个分量分成 4 组，每组 2 个，每组当作二维平面上的一个点：

```
pair 0:  (q_0, q_1)   → 视为 2D 平面上的点 p_0
pair 1:  (q_2, q_3)   → 视为 2D 平面上的点 p_1
pair 2:  (q_4, q_5)   → 视为 2D 平面上的点 p_2
pair 3:  (q_6, q_7)   → 视为 2D 平面上的点 p_3
```

现在我们有 4 个独立的二维点，可以各自在自己的平面里做旋转。

**等价的复数视角**：把每对 `(x, y)` 看成一个复数 `z = x + i·y`，一个 `d_head` 维实向量就变成了 `d_head/2` 个复数。复数乘以单位复数 `e^{iφ}` 在几何上就是旋转 φ 弧度，写起来特别干净：

$$z_i' = z_i \cdot e^{i\,m\,\theta_i}$$

##### (c) 两种配对约定（代码里要看清楚）

"哪两个分量算一对"有两种常见做法，**数学上等价，但权重要对齐**：

**方案 A：相邻配对（原论文 interleaved）**
```
索引：  0  1  2  3  4  5  6  7
配对：  [— —] [— —] [— —] [— —]
pair：    0     1     2     3
```

**方案 B：前后对半配对（LLaMA / HuggingFace "half-rotate"）**
```
索引：  0  1  2  3  |  4  5  6  7
          前半             后半
配对： (0,4) (1,5) (2,6) (3,7)
```

LLaMA、Qwen、DeepSeek 几乎都用方案 B。两种方案对 reshape 和权重列顺序有影响，实现时要统一。

##### (d) 每对做了什么：平面上转一个角度

对位置 `m` 的 token，把第 `i` 对 `(x_{2i}, x_{2i+1})` 绕原点**逆时针旋转 `m·θ_i` 弧度**：

$$\begin{pmatrix} x'_{2i}\\ x'_{2i+1}\end{pmatrix}
= \begin{pmatrix}\cos(m\theta_i) & -\sin(m\theta_i)\\ \sin(m\theta_i) & \cos(m\theta_i)\end{pmatrix}
\begin{pmatrix} x_{2i}\\ x_{2i+1}\end{pmatrix}$$

```
       y 轴
        │       ● 旋转后的点 (x', y')
        │     ⟋
        │   ⟋  ← 旋转 m·θ_i 弧度
        │ ⟋
        │⟋  ● 原点 (x, y)
   ─────●─────── x 轴
```

**关键性质：旋转保模长**。向量长度 `‖·‖` 旋转前后完全不变——RoPE **不改变 Q、K 的幅度信息，只改变方向**。而方向差就承担了编码位置差的工作。

##### (e) 为什么每对要用不同的频率 θ_i？

$$\theta_i = 10000^{-2i/d_\text{head}}$$

| 对的编号 i | θ_i（`d_head=128`） | 旋转速度 |
|---|---|---|
| 0 | `10000^0 = 1` | **最快**（每步位置转 1 弧度） |
| 1 | ≈ 0.93 | 快 |
| … | … | … |
| `d_head/2 - 1` | ≈ 1/10000 | **最慢**（几乎不动） |

**几何直觉：像一只多针钟表**——

- 秒针（高频对）转得飞快，能区分"第 5 个 vs 第 6 个 token"这种细粒度位置；
- 时针、日针（低频对）转得极慢，能区分"第 10 个 vs 第 1000 个 token"这种粗粒度位置。

**不同频率组合起来，模型既能辨别相邻位置、又能辨别远距离位置**。这与 Sinusoidal PE 用不同频率的正余弦是同一思想。

**关于 10000 这个 base**：原论文沿用 Transformer 原始 Sinusoidal PE 的取值，是经验选择。LLaMA 等模型做长上下文扩展时会调大它（如 500000、1000000），让"慢针"更慢，从而承载更长序列——这就是你可能听过的 **RoPE scaling / NTK-aware RoPE** 技术。

##### (f) 一个完全具体的小例子：`d_head = 4, m = 3`

设 `q = [1, 0, 0, 1]`，token 位置 `m = 3`，用方案 A 凑对：

```
pair 0: (1, 0),  θ_0 = 10000^{0}    = 1       → 旋转 3 × 1    = 3 弧度（≈172°）
pair 1: (0, 1),  θ_1 = 10000^{-2/4} = 0.01    → 旋转 3 × 0.01 = 0.03 弧度（≈1.7°）
```

应用旋转：

```
pair 0: x' = 1·cos(3) - 0·sin(3) ≈ -0.99
        y' = 1·sin(3) + 0·cos(3) ≈  0.14

pair 1: x' = 0·cos(0.03) - 1·sin(0.03) ≈ -0.03
        y' = 0·sin(0.03) + 1·cos(0.03) ≈  1.00
```

拼回：

```
q' ≈ [-0.99, 0.14, -0.03, 1.00]
```

**验证两点**：
1. 模长保持：`‖q‖ = √2 ≈ 1.414`，`‖q'‖ ≈ √(0.99² + 0.14² + 0.03² + 1²) ≈ 1.414` ✅
2. 前两维（高频对）被大幅旋转，后两维（低频对）几乎不动——高频编码细位置、低频编码粗位置的直观体现 ✅

##### (g) 整体记号

把所有 pair 的旋转矩阵拼成一个块对角矩阵 `R_m`（大小 `d_head × d_head`，对角上是 `d_head/2` 个 2×2 旋转块）：

$$R_m = \text{blkdiag}\!\left(R^{(\theta_0)}_m,\,R^{(\theta_1)}_m,\,\ldots\right)$$

对整个 Q、K 向量一次性作用：`Q' = R_m Q`，`K' = R_n K`。这是下一节 §4.4.3 里推导的起点。

#### 4.4.3 关键性质：相对位置自然涌现

把 RoPE 分别作用在 Q（位置 m）和 K（位置 n）上：

$$Q'_m = R_m\,Q_m,\quad K'_n = R_n\,K_n$$

注意力分数：

$$Q'^{\top}_m K'_n = Q_m^\top R_m^\top R_n K_n = Q_m^\top R_{n-m} K_n$$

（这里用到旋转矩阵的性质：`R_m^T R_n = R_{n-m}`——两次旋转的合成等于"差角"的一次旋转。）

**惊人的结论**：注意力分数**只依赖于相对位置 `n-m`**，而不依赖绝对位置。这正是我们想要的——"你关心的是两个 token 差多远，而不是它们各自在第几位"。

#### 4.4.4 伪代码

```python
def apply_rope(x, positions):
    # x: (L, d_head), positions: (L,)
    # 把 d_head 维切成偶数、奇数两半
    x_even, x_odd = x[..., 0::2], x[..., 1::2]     # 都是 (L, d_head/2)
    theta = 10000 ** (-torch.arange(0, d_head, 2) / d_head)  # (d_head/2,)
    angles = positions[:, None] * theta[None, :]    # (L, d_head/2)
    cos, sin = angles.cos(), angles.sin()
    x_even_new = x_even * cos - x_odd * sin
    x_odd_new  = x_even * sin + x_odd * cos
    return interleave(x_even_new, x_odd_new)        # (L, d_head)

Q = apply_rope(Q, positions)   # Q 被就地旋转
K = apply_rope(K, positions)   # K 被就地旋转
# V 不做 RoPE；注意力后续照常算
```

**注意：RoPE 只作用在 Q 和 K 上，不作用在 V 上。**

#### 4.4.5 为什么 RoPE 会和 MLA 的权重吸收打架？

现在带着这个理解回到 MLA。回忆 §4.3.2 的权重吸收：

$$\text{score}_i = c_Q\,\widetilde{W}_i\,c_{KV}^\top,\quad \widetilde{W}_i = W_i^{UQ}(W_i^{UK})^\top$$

能吸收的核心前提是 **`W̃_i` 是一个固定不变、和位置无关的矩阵**。

但如果我们对 Q、K 应用 RoPE：

$$\text{score}_i = Q_m^{\prime\top}K'_n = (R_m Q_m)^\top (R_n K_n) = Q_m^\top R_{n-m} K_n$$

中间多了一个 **`R_{n-m}`——它依赖于每一对 (m, n) 的具体位置差**！这意味着：

- 对序列中每一对 token 位置，中间的旋转矩阵都不一样；
- 我们**没法**像权重吸收那样"推理前预先算一次就够用"；
- 如果硬要吸收，每个 (m, n) 对都要单独算一个矩阵——直接放弃了权重吸收的好处。

**这就是 §4.5（原 §4.4）要讲的"RoPE 与 MLA 不兼容"的数学本质。** DeepSeek 的解法"解耦 RoPE"就是把 Q/K 切成"可吸收的非 RoPE 部分"和"不可吸收但维度很小的 RoPE 部分"，分头处理——下一节详细讲。

### 4.5 有了 RoPE 之后 MLA 怎么做？——解耦 RoPE（Decoupled RoPE）

§4.4.5 已经指出：RoPE 引入的 `R_{n-m}` 让权重吸收失效。本节详细讲 DeepSeek 是怎么解决这个矛盾的。

#### 4.5.1 第 1 步：把冲突写成数学式

没有 RoPE 时（§4.3.2）：

$$\text{score}_i = c_Q\,\underbrace{W_i^{UQ}(W_i^{UK})^\top}_{\widetilde{W}_i \text{ 是常量}}\,c_{KV}^\top$$

中间是**位置无关的固定矩阵**，可以预算好。

加上 RoPE 后：

$$\text{score}_i = (R_m Q_i)^\top (R_n K_i) = Q_i^\top R_{n-m} K_i = c_Q\,W_i^{UQ}\,R_{n-m}\,(W_i^{UK})^\top\,c_{KV}^\top$$

**中间多了一个 `R_{n-m}`，它依赖具体的 (m, n) 对，每对 token 都不一样。** 因此无法"推理前预算一次 `W̃_i`"——这就是根本冲突。

#### 4.5.2 第 2 步：几个"想当然"的方案为什么不行

- **方案 1：把 RoPE 作用在 `c_{KV}` 上** → 破坏"相对位置自然涌现"的性质（`R_m^T R_n = R_{n-m}` 不再成立）；
- **方案 2：把 `R_{n-m}` 吸收进权重** → 动态 vs 静态矛盾，根本不可能；
- **方案 3：不用 RoPE 改用 ALiBi** → 放弃 RoPE 的长上下文外推、生态兼容等优势。

#### 4.5.3 第 3 步：DeepSeek 的解法——"分工"而不是"吸收"

> **既然 RoPE 和权重吸收不能共存，那就把 Q、K 的维度拆成两块：**
> - **内容块（Content / "nope" = no positional encoding）**：走低秩压缩 + 权重吸收，**不加 RoPE**；
> - **位置块（RoPE block）**：专门承载位置信息，**加 RoPE**，不做吸收。
>
> **注意力分数 = 两块各自点积的和。**

每个头的 Q、K 变成拼接向量：

$$Q_i = [\,Q_i^{C}\;;\;Q_i^{R}\,],\qquad K_i = [\,K_i^{C}\;;\;K^{R}\,]$$

**关键技巧**：`K^R` 没有 `_i` 下标——它在所有头之间**共享**（类似 MQA 风格）！这是为了把 RoPE 部分的缓存开销压到最低。

#### 4.5.4 第 4 步：具体 shape 走一遍（DeepSeek-V2 真实配置）

设：
- `d_model = 5120, h = 128`
- `d_h_nope = 128`（每头 content 部分维度，走吸收）
- `d_h_rope = 64`（每头 RoPE 部分维度）
- `d_c = 512`（KV 压缩潜在维度），`d_c' = 1536`（Q 压缩潜在维度）

每个头 Q、K 的总维度 = `d_h_nope + d_h_rope = 128 + 64 = 192`。

**K 侧：两条路径并行**

路径 1：content K（per-head，走压缩 + 吸收）

```
x   (5120)   → c_{KV} = x · W^{DKV}     (512)     ← 缓存
c_{KV}  × W_i^{UK}  (512, 128)  →  K_i^C  (128)   ← 不加 RoPE
                                                   （W_i^{UK} 将被吸收，实际不真算）
```

路径 2：shared RoPE K（所有头共享，MQA 风格）

```
x   (5120)   × W^{KR}  (5120, 64)  →  k^R_raw  (64)
apply_rope(k^R_raw, position n)    →  K^R      (64)   ← 缓存
```

每个头的 K：`K_i = [K_i^C (128); K^R (64)]`（维度 192）。

**Q 侧：类似两条路径**

```
c_Q = x · W^{DQ}        (1536)
Q_i^C = c_Q · W_i^{UQ}  (1536 → 128)             ← 不加 RoPE
Q_i^R = apply_rope(c_Q · W_i^{QR}, position m)   ← 加 RoPE，per-head
Q_i   = [Q_i^C (128); Q_i^R (64)]   (192)
```

**V 侧：完全不受影响**

V 不参与 RoPE（RoPE 只作用在 Q、K 上），所以 V 的生成照旧：

$$V_i = c_{KV}\,W_i^{UV}$$

依然可以按 §4.3.3 把 `W^{UV}` 吸收到 `W^O` 里。

#### 4.5.5 第 5 步：注意力分数怎么算——见证奇迹

拼接向量的点积等于两段各自点积之和：

$$\boxed{\;\text{score}_i \;=\; \underbrace{Q_i^C \cdot (K_i^C)^\top}_{\text{content 项}} \;+\; \underbrace{Q_i^R \cdot (K^R)^\top}_{\text{RoPE 项}}\;}$$

**两项的计算特点完全不同：**

**content 项**——完美吸收，和纯 MLA 一样：

$$Q_i^C\,(K_i^C)^\top = c_Q\,W_i^{UQ}(W_i^{UK})^\top c_{KV}^\top = c_Q\,\widetilde{W}_i\,c_{KV}^\top$$

`W̃_i` 可预先算好，全程不物化 `K_i^C`。

**RoPE 项**——直接点积，不吸收：

- `Q_i^R` 维度仅 64（per-head），`K^R` 维度仅 64（跨头共享一份）；
- 维度小，直接算就行，也无法吸收（因为带 RoPE）；
- `K^R` 跨头共享进一步省缓存和带宽。

**两项加起来就是完整的注意力分数**——RoPE 的位置感知保留，权重吸收的效率也保留。

#### 4.5.6 第 6 步：KV Cache 最终账单

每 token 每层缓存两样东西：

| 量 | Shape | 数值 |
|---|---|---|
| `c_{KV}`（共享压缩 KV） | `(d_c,)` | 512 |
| `K^R`（共享 RoPE K） | `(d_h_rope,)` | 64 |
| **合计** | | **576** |

横向对比：

| 方法 | 每 token 每层 | 相对 MHA |
|---|---|---|
| MHA | 32,768 | 100% |
| MQA | 256 | 0.78% |
| GQA (g=8) | 2,048 | 6.25% |
| **MLA（解耦 RoPE）** | **576** | **1.76%** |

MLA 的缓存开销和 MQA 同量级，但效果接近甚至好于 MHA——这就是解耦 RoPE 带来的工程胜利。

#### 4.5.7 第 7 步：推理一步的完整流水图

```
                        ┌────────────────────────────────────────────┐
                        │  缓存（每 token 每层）                      │
                        │    c_{KV}: 512 维   |   K^R: 64 维（共享）  │
                        └────────────────────────────────────────────┘
                                         ▲
                    ┌───────── 当前 token 的 x ─────────┐
                    │                                    │
    ┌───────────────┴──────────────┐      ┌──────────────┴──────────────┐
    │ 更新 cache：                  │      │ 算当前 Q：                   │
    │  c_{KV}^{new} = x · W^{DKV}   │      │  c_Q = x · W^{DQ}           │
    │  K^R_{new} = RoPE(x·W^{KR},n) │      │  Q_i^C = c_Q · W^{UQ}_i     │
    └───────────────┬──────────────┘      │  Q_i^R = RoPE(c_Q·W^{QR}_i,m)│
                    │                     └──────────────┬──────────────┘
                    ▼                                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  每头分数 = Q_i^C · W̃_i · c_{KV}^T   ←  权重吸收（全压缩空间）│
        │            +  Q_i^R · (K^R)^T        ←  小维度普通点积       │
        └─────────────────────────────────────────────────────────────┘
                                         ▼
                            softmax → A_i → 聚合 V
                                         │
                       V 侧走 Ŵ_i^O = W^{UV}_i · W^O_i 吸收（§4.3.3）
                                         ▼
                                  输出 MHA(x)
```

#### 4.5.8 一句话总结

> **解耦 RoPE = 把每个头的 Q、K 拆成「内容块 + 位置块」：内容块不带 RoPE，走低秩压缩 + 权重吸收；位置块带 RoPE，且 K 的这一小部分跨所有头共享。两块点积之和就是最终注意力分数——鱼和熊掌兼得。**

### 4.6 代价

- 结构比 GQA 复杂，实现门槛更高；
- 不能像 GQA 那样从旧 MHA checkpoint 直接 uptrain，需要从头训或大规模继续训练；
- 但一旦训好，**KV Cache 可以比 GQA 还小几倍，效果反而更好**（DeepSeek-V2 报告中优于 MHA 基线）。

---

## 5. DSA（DeepSeek Sparse Attention，DeepSeek 稀疏注意力）

DeepSeek 在 2025 年 9 月的 DeepSeek-V3.2-Exp 中提出，可以看作 **MLA 的下一代演进**：MLA 解决"每 token 存多少"（显存），DSA 解决"每步看多少 token"（算力）。二者正交，叠加使用。

### 5.1 动机：MLA 解决了显存，但没解决算力

MLA 把每 token KV Cache 压到 ~576 维，但还有一个问题没动：

> **注意力本身的计算量仍是 `O(L²)`**。

每个新 token 作为 Query，要和**前面所有 token**的 Key 做点积。即使 Key 从 cache 里读很便宜，算 Q·K 本身的 FLOPs 也随 L 线性涨，整条序列生成就是 L² 级。

- L = 4K：还能接受；
- L = 128K：**计算量是 L=4K 的 1024 倍**；
- L = 1M：基本跑不动。

### 5.2 为什么稀疏注意力历史上不太成功？

"砍算力"的思路叫 **Sparse Attention**：每个 query 只看一部分 key。早期尝试：

| 方法 | 选择策略 | 问题 |
|---|---|---|
| Longformer | 固定"局部窗口 + 少量全局 token" | 规则死板，错过远处真正相关的 token |
| BigBird | 局部 + 全局 + 随机 | 同上，随机未必命中 |
| Reformer | LSH 桶近似 | 近似误差大，工程复杂 |
| Sliding Window（Mistral） | 只看最近 N 个 token | 长程依赖丢失 |

**共同痛点**：用**固定规则或近似算法**决定"谁重要"，但真正的重要性是**内容相关、动态变化**的。

> **DSA 的突破口**：让模型**学会**"谁重要"，每个 query 都动态挑选它真正关心的 top-k 个 token。

### 5.3 核心组件：Lightning Indexer（闪电索引器）

DSA 引入一个**轻量级的"索引器"**——一个简化版注意力模块，专门给每个历史 token 打"相关度分"，然后选 top-k。

**设计要点**：

- **头数少**（比如 64），远少于主注意力（128）；
- **每头维度小**（比如 128），可用 FP8 存储进一步省算力；
- **只算分数，不聚合 V**——不参与最终输出，只做"路由"。

**计算过程**：对当前 query 位置 `t` 和历史位置 `s`（`s < t`）：

$$q_t^I = x_t\,W^{IQ},\quad k_s^I = x_s\,W^{IK},\quad I(t, s) = q_t^I \cdot (k_s^I)^\top$$

对所有 `s ∈ {1, ..., t-1}` 算一遍 `I(t, s)`。这一步算力开销是 `O(L · d_I)`，`d_I` 很小，比主注意力的 `O(L · h · d_head)` 便宜得多。

### 5.4 Top-k 选择：细粒度路由

拿到所有分数后：

```
按 I(t, s) 排序，选前 k 个 s，记作集合 S_t ⊂ {1, ..., t-1}
```

**典型配置**：k = 2048（远小于 L = 128K）。

主注意力只在 `S_t` 上做：

$$\text{out}_t = \text{MLA-Attention}\!\left(Q_t,\,\{K_s, V_s\}_{s \in S_t}\right)$$

**算力从 `O(L)` 降到 `O(k)`——常数级别，与序列长度无关！**

**关键特性：per-query 细粒度**

不是"整段共用一个稀疏模式"，而是**每个 query 独立选自己的 top-k**：

- Token A（生成代码注释时）可能挑到很远的函数定义；
- Token B（生成循环体时）可能只看最近几行；
- 完全不同——**这才是"学习出来的动态稀疏"的威力**。

### 5.5 训练 Indexer：两阶段策略

Indexer 必须打分准，否则挑错 token 会严重影响质量。DeepSeek 的做法是两阶段训练。

**阶段 A：Dense Warm-up（密集热身）**

- 仍用**完整密集注意力**训练主模型；
- **同时训练 Indexer**，让它的分数分布拟合真实注意力分布；
- 损失加一项 KL 散度：

  $$\mathcal{L}_\text{index} = \text{KL}\big(\text{softmax}(A_t)\;\|\;\text{softmax}(I(t,\cdot))\big)$$

  用大模型老师的注意力当监督信号——一种自蒸馏式训练。

**阶段 B：Sparse Fine-tune（稀疏微调）**

- Indexer 已能较好模拟真实注意力后；
- 切换到**真正稀疏模式**：主注意力只算 top-k；
- 让主模型适应"我只能看 k 个 token"。

**两阶段避免了冷启动问题**——一开始就稀疏会因 Indexer 未学好而训练崩溃。

### 5.6 完整架构图

```
                     输入序列  x_1, x_2, ..., x_t
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
      ┌─────────────┐       ┌─────────────┐         ┌──────────────┐
      │  Indexer    │       │   MLA 压缩  │         │   MLA 查询   │
      │             │       │             │         │              │
      │ q_t^I, k^I  │       │   c_{KV}    │         │  c_Q, Q_i    │
      │ (小维度)     │       │  (全 token) │         │              │
      └─────┬───────┘       └──────┬──────┘         └──────┬───────┘
            │                      │                        │
            ▼                      │                        │
      每个 s 算 I(t,s)              │                        │
            │                      │                        │
            ▼                      │                        │
    ┌────────────────┐             │                        │
    │  选 top-k 个 s  │ → 集合 S_t  │                        │
    └────────┬───────┘             │                        │
             │                     │                        │
             └──────────→ 只从 c_{KV} 中取 S_t 对应的行 ◄─────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ MLA Attention (只在 k)│
                        │  权重吸收 + 解耦 RoPE │
                        └──────────┬───────────┘
                                   ▼
                                 输出 y_t
```

### 5.7 收益量化（以 DeepSeek-V3.2-Exp 为例，L=128K, k=2048）

| 量 | 标准 MLA | DSA |
|---|---|---|
| 主注意力 FLOPs（每 token） | `O(L·h·d_head)` = 100% | `O(k·h·d_head)` ≈ **1.6%** |
| Indexer FLOPs（每 token） | — | `O(L·d_I)`，很小 |
| KV Cache | 576/token/layer | 576/token/layer（不变） |
| 额外存储 | — | Indexer keys（小） |

**实际收益**：V3.2-Exp 发布时 API 价格直接砍半，长上下文推理成本大幅下降，公开榜单效果几乎不降。

### 5.8 DSA 和前几代的关系

| 代次 | 解决的问题 | 核心手段 |
|---|---|---|
| MHA | 表达力 | 多头独立 Q/K/V |
| MQA | 缓存显存 | 多头共享 K/V |
| GQA | MHA/MQA 折中 | 分组共享 K/V |
| **MLA** | **进一步压缩缓存** | **低秩潜在压缩 + 权重吸收 + 解耦 RoPE** |
| **DSA** | **长上下文算力** | **MLA + 学习式稀疏（Lightning Indexer + top-k）** |

**DSA 不取代 MLA，而是在 MLA 之上叠加一层路由机制：MLA 管显存，DSA 管算力，两者正交叠加。**

### 5.9 代价与限制

1. **训练更复杂**：两阶段训练、额外的 Indexer 参数和 KL 损失；
2. **Indexer 本身要推理**：虽然便宜但是额外开销，短序列下可能不值；
3. **kernel 实现门槛高**：per-query 动态 top-k 对 GPU kernel 非常不友好，需要专用稀疏注意力 kernel；
4. **效果非零损失**：极长依赖、罕见但关键的 token 可能被漏选——需仔细调 k 和 Indexer 容量。

DSA 主要在**长上下文（≥32K）场景**下开启收益最明显。短序列继续用标准 MLA 更划算。

### 5.10 一句话总结

> **DSA = MLA（缓存已压缩）+ Lightning Indexer（学习式 top-k 路由）**
>
> **把注意力从"每 token 看所有 token"变成"每 token 只看它真正关心的 k 个 token"，k 是常数而非 O(L)——长上下文的算力瓶颈被打穿。**

---

## 6. 对比总结

### 6.1 一张表看清区别

| 维度 | MHA | MQA | GQA | MLA | DSA |
|---|---|---|---|---|---|
| 提出时间 | 2017 | 2019 | 2023 | 2024 | 2025 |
| Q 头数 | h | h | h | h | h |
| K/V 结构 | h 组独立 | 1 组共享 | g 组共享 | 压缩成 `d_c` 维潜在向量 | 同 MLA + top-k 路由 |
| 每 token KV Cache | 100% | 1/h | g/h | ~1.7% | ~1.7%（同 MLA，加小 Indexer key） |
| 注意力算力 | O(L) | O(L) | O(L) | O(L) | **O(k)，k 为常数** |
| 表达能力 | 最强 | 明显下降 | 接近 MHA | ≈ 或 > MHA | ≈ MLA |
| 训练兼容性 | — | 需重训 | 可从 MHA uptrain | 需重训/大改 | 需两阶段训练 |
| 实现复杂度 | 最简单 | 简单 | 简单 | 复杂（RoPE 解耦） | 最复杂（稀疏 kernel） |
| 代表模型 | 原始 Transformer, GPT-2/3, LLaMA-1 | PaLM, Falcon | LLaMA-2/3, Mixtral, Qwen2 | DeepSeek-V2/V3/R1 | DeepSeek-V3.2-Exp |

### 6.2 演化脉络（一句话串起来）

> **MHA** 表达力强但 KV Cache 太大 → **MQA** 激进共享 K/V 省显存但掉点 → **GQA** 分组折中是工程最优 → **MLA** 换思路，用低秩潜在压缩，既省又不掉点 → **DSA** 在 MLA 之上加 learn-to-route，把算力也砍到常数级。

### 6.3 直观类比（帮助记忆）

想象 h 个人（多头）要查同一本图书馆索引（K/V）：

- **MHA**：每个人有自己**一整套**索引；全但重。
- **MQA**：所有人共用**一套**索引；轻但粗。
- **GQA**：分几个小组，每组一套索引；权衡。
- **MLA**：大家共用一个**压缩过的索引摘要**，查的时候按需展开；又轻又准。
- **DSA**：在 MLA 基础上再加一个"**小助理（Indexer）**"，每次先扫一眼帮你挑出最相关的 k 本书，你只翻这 k 本——不用每次都看完整个图书馆。

### 6.4 选择建议（如果你在设计模型）

- 想要最强效果、不在乎推理显存 → **MHA**
- 极端追求推理速度、能承受效果下降 → **MQA**
- 想在现有 MHA 模型上低成本升级 → **GQA**（目前工业界首选）
- 从零训练、目标长上下文 / 高并发 → **MLA**
- 超长上下文（≥32K）、追求推理成本极致 → **DSA**

---

## 7. 进一步阅读

- **MHA**: Vaswani et al., *Attention Is All You Need*, 2017.
- **MQA**: Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019.
- **GQA**: Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
- **MLA**: DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024.
- **DSA**: DeepSeek-AI, *DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention*, 2025.

---

## 8. 给学生的思考题

1. 假设一个模型 `d_model=4096, h=32, d_head=128, n_layers=32`，序列长度 8192，batch=1，fp16。请分别估算 MHA、GQA（g=8）、MQA 的 KV Cache 显存。
2. 为什么 GQA 能从 MHA checkpoint "平均合并"得到而效果不太掉？背后的数学直觉是什么？
3. MLA 的"权重吸收"技巧为什么只在**推理**阶段适用？训练时为什么必须真正物化出 K、V？
4. 如果要把 GQA 模型改造成 MLA，最大的工程挑战会在哪里？（提示：位置编码）
5. DSA 的 Lightning Indexer 为什么用 KL 散度而不是直接用交叉熵去拟合主注意力？如果在稀疏微调阶段仍然保留 KL 损失，会有什么影响？
6. DSA 的 top-k 选择在训练时是可微的吗？如果不可微，反向传播怎么走？（提示：straight-through estimator）

有想法的话可以回来继续讨论。
