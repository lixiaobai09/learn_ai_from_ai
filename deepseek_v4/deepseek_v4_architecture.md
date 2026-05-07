# DeepSeek-V4 新架构知识点详解

> 写给大一计算机新生的学习笔记。  
> 原始材料：[DeepSeek-V4 技术报告](./DeepSeek_V4.pdf)（44 页正文 + 14 页附录，共 58 页）  
> 标题：*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
>
> **学习目标**：搞懂 V4 相对 V3 / V3.2 究竟"新"在哪里，以及为什么新。
>
> **学习方法建议**：先通读"全景图"和"知识地图"，建立宏观认知；然后挑你最感兴趣的一个章节深读，再逐步扩展。**不要试图一次理解所有细节**——技术报告本来就是给已经会一半的人看的。

---

## 0. 写在最前：先看"WHY"，再看"HOW"

学一门新架构最常见的坑，是直接钻进公式细节，结果学到最后只记住了一堆"它做了 X"，却不知道"为什么要做 X"。

DeepSeek-V4 这篇论文要解决的**根本问题**只有一句话：

> **当前所有 Transformer 模型在"超长上下文"（比如 1M = 100 万 tokens）下都太贵了。**

为什么贵？因为标准注意力的算力是 **O(n²)** 的——序列长度翻 10 倍，注意力的计算量翻 100 倍，KV Cache 显存也翻 10 倍。100 万 tokens 意味着注意力部分要算 10¹² 次乘加，这是单卡根本扛不住的。

V4 的所有创新，几乎都是围绕这一个目标展开：**在不损失能力的前提下，把"超长上下文"变得便宜**。

它最终交出的答卷：

| 场景：1M tokens 单 token 推理 | KV Cache 大小 | 计算量 (FLOPs) |
|---|---|---|
| DeepSeek-V3.2（基线）           | 100% | 100% |
| DeepSeek-V4-Pro（49B 激活）     |  10% |  27% |
| DeepSeek-V4-Flash（13B 激活）   |   7% |  10% |

> 翻译成人话：**V4-Flash 用 1/10 的算力和 1/14 的显存，在 100 万 token 上下文里跑出了不输甚至更好的效果。**

---

## 1. 全景图：V4 系列两个模型

V4 是一个**系列**，不是一个模型。它包含：

| 模型 | 总参数 | 激活参数 (每 token) | 层数 | 隐藏维度 | 路由专家数 | 激活专家数 |
|---|---|---|---|---|---|---|
| **DeepSeek-V4-Flash** | 284B  | 13B | 43 | 4096 | 256 | 6 + 1 共享 |
| **DeepSeek-V4-Pro**   | 1.6T  | 49B | 61 | 7168 | 384 | 6 + 1 共享 |

> 📌 **MoE 小知识（铺垫给后面）**：MoE = Mixture of Experts。"总参数 1.6T"是说模型一共有 1.6 万亿参数，但**每个 token 只激活其中 49B**（约 3%）。这就像一个有 384 位专科医生的医院，每个病人只挂 6 位医生的号，而不是让所有医生都来诊断。算力按"激活参数"算，知识容量按"总参数"算——这是 MoE 的基本魅力。

两个模型还各自有 3 种"思考模式"：

- **Non-think**：直接回答，快但浅。
- **Think High**：`<think>...</think>` 内输出推理过程，中等深度。
- **Think Max**：注入特殊 system prompt，把推理 token 数推到最大（拼命想）。

V4-Pro 的最强模式叫 **V4-Pro-Max**——这是论文里反复对标 GPT-5.4 / Claude Opus 4.6 / Gemini 3.1 Pro 的那一档。

---

## 2. 知识地图：V4 到底新在哪？

V4 = **V3.2 的架构骨架** + **5 个核心新东西** + **一堆工程优化**。

```
                  DeepSeek-V4 架构创新地图
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   继承自 V3 (没变):                                      │
│   ├── DeepSeekMoE (细粒度路由专家 + 共享专家)            │
│   ├── MTP (Multi-Token Prediction，多 token 预测)        │
│   └── Transformer Block 整体框架                         │
│                                                          │
│   ★ 全新创新 (V4 核心):                                  │
│   ├── ① mHC      (Manifold-Constrained Hyper-Connections)│
│   │       —— 强化残差连接，让深网络更稳                  │
│   ├── ② CSA + HCA 混合注意力                             │
│   │       —— 长上下文降本的主力武器                      │
│   ├── ③ Muon 优化器 (替代 AdamW 主体)                    │
│   │       —— 更快收敛、更稳训练                          │
│   ├── ④ FP4 量化感知训练 (QAT)                           │
│   │       —— 推理省显存                                  │
│   └── ⑤ On-Policy Distillation (替代混合 RL)             │
│           —— 后训练中"多专家融合"的新范式                │
│                                                          │
│   ★ 重要工程基础设施:                                    │
│   ├── MegaMoE (通信-计算融合 mega-kernel)                │
│   ├── TileLang (kernel 开发 DSL，集成 Z3 SMT 求解器)     │
│   ├── 批不变 + 确定性内核                                │
│   ├── 异构 KV Cache 布局 + 磁盘缓存                      │
│   └── DSec (Agent 沙箱平台)                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

下面我按"由浅入深"的顺序拆解。建议你**先读 §3、§4（容易），再读 §5、§6（核心，注意力和优化器），再读 §7（工程，对应你以后的并行计算课）**。

---

## 3. 继承自 V3 的"地基"（10 分钟回顾）

V4 没有重新发明轮子，下面这些直接沿用 V3：

### 3.1 DeepSeekMoE：细粒度专家 + 共享专家

普通 MoE 一般有 8 ~ 16 个大专家，每个 token 激活 2 个。  
DeepSeekMoE 的做法是把专家**切得更细**（256 ~ 384 个），但每个更小，每个 token 激活 6 个，**外加 1 个所有 token 都用的"共享专家"**。

**直觉理解**：
- 细粒度 → 专家分工更精细，比如可以分出"Python 专家"、"中文写作专家"，而不是一个笼统的"代码 + 文本专家"。
- 共享专家 → 处理那些"所有任务都要会"的通用知识（语法、常识），避免在每个专家里重复学习。

**V4 的小调整**：
1. 路由打分函数从 V3 的 `Sigmoid` 改成 `Sqrt(Softplus(·))`——这是个小调味品，目的是让分数分布更平滑。
2. 前 3 层 MoE 用 **Hash 路由**（按 token ID 直接 hash 到固定专家），不用学习。这是为什么？因为前几层学到的是低层特征（类似词形、词性），路由本身没什么"智力"，省掉学习的开销。
3. 取消了 V3 里"每个 token 最多路由到 N 个机器节点"的硬约束——重新设计了并行策略来弥补带宽。

### 3.2 MTP（Multi-Token Prediction）

普通 LLM 一次只预测下一个 token。MTP 让模型**一次预测后面 D 个 token**（V4 里 D=1，相对保守）。

**为什么有用**：
- 训练信号更密集：每个位置不止学"下一个词"，还学"再下一个词"，相当于训练目标更难，模型学得更深。
- 推理时还能用作"投机解码（speculative decoding）"加速。

V4 里 MTP 的实现完全沿用 V3，没改。

> ✅ **小结**：MoE + MTP 这两件事 V4 几乎没动，是它的"地基"。后面的所有创新都是搭在这个地基上的。

---

## 4. 创新 ①：mHC——增强版的残差连接

### 4.1 你已经会的：标准残差连接

Transformer 每层的核心都是这一行：

```
x_{l+1} = x_l + F_l(x_l)
```

这个 `+` 就是**残差连接**（ResNet 那个 skip connection），它解决的是"深网络梯度消失"的问题。

### 4.2 进阶：Hyper-Connection (HC)

标准残差只用一个"通道"传信号。HC 的想法是：**搞 n_hc 条平行的残差通道**，让信号在多条通道间混合，每层结束时再"合流"到 n_hc 个通道里。

写成公式：

```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)
```

其中 X_l 不再是 d 维向量，而是 **n_hc × d 的矩阵**（n_hc 条通道，每条 d 维）。  
A、B、C 是三个学习矩阵，控制不同通道之间的混合方式。

**好处**：相当于给残差加了一个新的"扩展维度"——除了"宽度 d"和"深度 L"，又多了一个"残差通道数 n_hc"可以扩展模型容量。

**问题**：HC 论文的作者发现，**一旦 L 大了，训练经常炸**——B_l 的特征值可能变得 > 1，信号每过一层就被放大一点点，多层下来就指数爆炸。

### 4.3 V4 的创新：mHC（Manifold-Constrained）

DeepSeek 团队的做法是：**强制 B_l 矩阵必须是"双随机矩阵（doubly stochastic matrix）"**——即每行每列的和都等于 1，所有元素非负。

**为什么这能解决爆炸？**

数学事实：双随机矩阵的**谱范数（最大奇异值）一定 ≤ 1**。也就是说，X 经过 B_l 之后，"长度"绝不会变大，只会保持或缩小。这就把"信号不能爆炸"这件事**从经验技巧变成了数学保证**。

**直觉类比**：标准残差是"一条河直接流过去"；HC 是"分流成多条河，到处乱流"；mHC 是"虽然分流，但保证总水量不变（守恒）"。守恒就稳了。

**怎么把一个普通矩阵投影成双随机矩阵？**

用经典的 **Sinkhorn-Knopp 迭代**：交替对每一行、每一列做归一化，重复 t_max=20 轮。这是 OT（最优传输）领域的老朋友。

```
M^(0) = exp(B̃_l)              # 先取 exp 保证非负
loop t = 1 ... 20:
    M^(t) = 行归一化(列归一化(M^(t-1)))
B_l = M^(t_max)
```

迭代收敛后，B_l 就是合法的双随机矩阵。

**V4 的具体配置**：n_hc = 4（4 条残差通道），t_max = 20 轮 Sinkhorn 迭代。

### 4.4 工程难点：mHC 不便宜

引入 mHC 会让**激活内存**和**流水线通信**都增加。V4 的应对：

1. 给 mHC 写**融合 kernel**（把多个小算子合成一个大算子，减少访存）。
2. 用**重计算（recomputation）策略**：反向传播时不存中间激活，重新算一遍。
3. 调整 DualPipe 1F1B 流水线调度，把 mHC 的通信和计算并行起来。

最终把 mHC 的额外 wall-time 开销控制在了流水线 stage 的 **6.7%**——基本可接受。

> ✅ **mHC 一句话总结**：用"双随机矩阵约束"给多通道残差加了一道安全锁，让深层网络的训练更稳。

---

## 5. 创新 ②：CSA + HCA 混合注意力（重头戏）

这是 V4 的**核心王牌**。如果你只想搞懂一个东西，就搞懂这个。

### 5.1 先看为什么标准注意力这么贵

回顾一下标准注意力（你应该在 `flash_attn_1` 笔记里见过）：

- 每个 query 要和**所有**前面的 key 算点积 → O(n) 计算
- 全部 query 这么干 → **O(n²)** 总计算
- KV Cache 要存**每个 token 的每一层**的 K 和 V → 显存随 n 线性增长，但乘上层数和头数后非常大

n = 1M 时这两件事都崩。

### 5.2 V3.2 已经做了什么（先讲铺垫）

V3 用 **MLA（Multi-head Latent Attention）**：把 KV 投影到低维 latent 再展开，KV cache 缩水。  
V3.2 在 MLA 基础上加了 **DSA（DeepSeek Sparse Attention）**：每个 query 只挑 top-k 个最相关的 KV 来算注意力，**让 O(n²) 变 O(n·k)**。

→ 这些你的 `mha_gqa_mqa_mla_dsa/attention_variants.md` 笔记里应该已经有了。

### 5.3 V4 的两板斧：CSA 和 HCA

V4 的洞察：**只稀疏不够，还要压缩；只压缩不够，还要分场景压**。

它把注意力层分成两类，**交替使用**（interleaved）：

| 类型 | 全称 | 压缩比 | 是否稀疏 | 适用场景 |
|---|---|---|---|---|
| **CSA** | Compressed Sparse Attention      | m = 4 (温和)   | ✅ 用 DSA top-k | 需要"既看细节又看全局" |
| **HCA** | Heavily Compressed Attention     | m' = 128 (激进) | ❌ 稠密 (但只对压缩后) | 主要看大局 |

**为什么要两种交替？**——直觉上：
- HCA 像"先把整本书每章压成一句摘要"，跨章节理解全局，但丢失了章内细节。
- CSA 像"把书每 4 段压成一段，再从所有段里挑最相关的看"，保留细节，但要做选择。
- **交替使用**就让模型既有"全局视角"又有"局部精读"。

> 类比：你期末复习时，会先翻"知识树/思维导图"（HCA：极致压缩），再针对重点章节"逐节细看"（CSA：温和压缩 + 挑重点）。两件事配合，才能在有限时间里复习好。

### 5.4 CSA 详解（先看简单的总图）

```
            ┌─────────────────────────┐
            │  原始 KV tokens (n 个)  │
            └────────────┬────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │ Token-level Compressor          │
        │ 每 m=4 个 token 压成 1 个 entry │
        └───────┬────────────────────┬───┘
                │                    │
          ┌─────▼─────┐       ┌──────▼──────┐
          │ Indexer K │       │ Compressed  │
          │ (压缩后)  │       │ KV Entries  │
          └─────┬─────┘       │ (n/m 个)    │
                │             └──────┬──────┘
                ▼                    │
          ┌──────────┐               │
          │ Lightning│  ←  Indexer Q │
          │ Indexer  │               │
          └─────┬────┘               │
                │                    │
                ▼                    │
        Top-k 选 k 个最相关          │
                │                    │
                └────────┬───────────┘
                         ▼
                ┌──────────────────────┐
                │ 选中的 k 个压缩 KV   │+ 滑窗 KV (n_win)
                └──────────┬───────────┘
                           │
                           ▼
              Multi-Query Attention
                  (核心注意力)
                           │
                           ▼
                Grouped Output Projection
                           │
                           ▼
                       输出 ô
```

**三个关键步骤**：

#### 步骤 1：Token 级压缩（n → n/m）

输入是 n 个 token 的 hidden state H ∈ ℝ^(n×d)。  
对**每 m = 4 个相邻 token**，用一组**学习权重 + 位置 bias** 做加权平均，压成 1 个 KV entry。

公式（简化版，去掉两套 ab 结构）：

```
S_{i*m : (i+1)*m} = Softmax(Z_{i*m : (i+1)*m} + B)   # 算压缩权重
C_i^Comp = Σ_j S_j ⊙ C_j                              # 加权汇总成 1 个 entry
```

直觉：4 个 token 就像 4 个发言人，模型学一个"主持人"决定每个人话语的权重，最后压成一个"会议纪要 entry"。

> 论文里其实有 a/b 两套并行的压缩通道（看 PDF 公式 9-12），保证信息覆盖更全，但学一遍后回头看就懂了。

#### 步骤 2：Lightning Indexer 选 top-k

压缩后还有 n/m 个 entry。如果 n=1M，m=4，那还有 25 万个 entry——还是太多。

所以 V4 复用 V3.2 的 **DSA top-k**：让每个 query 只挑 **k = 512（Flash）/ 1024（Pro）** 个最相关的 entry 来算注意力。

但"挑相关的"本身要打分，这个打分网络叫 **Lightning Indexer**：
- Query 端：低秩投影（先压到 d_c 维再展开成 n_I_h=64 个 indexer query）
- Key 端：用前面的压缩 KV entry 的 indexer 版本
- 打分公式（公式 16）：

```
I_{t,s} = Σ_h w^I_{t,h} · ReLU(q^I_{t,h} · K^IComp_s)
```

注意这里用了 **ReLU 而不是 softmax**——是为了让打分能直接做 top-k 比较，且 indexer 可以用 **FP4** 跑（极低精度），便宜得不行。

#### 步骤 3：Shared KV MQA + 分组输出投影

挑出来的 k 个 entry 同时当 key 也当 value（MQA：Multi-Query Attention，所有 query head 共享同一组 KV）。

最后的输出投影做了一个**分组优化**：因为 V4 的总头维度 c·n_h 很大，直接投到 d 维太贵。所以先把 n_h 个头分成 g 组，每组先投到一个中间维度 d_g，再拼起来投到 d。

这是经典的"低秩分解 / bottleneck"省算力技巧。

### 5.5 HCA 详解（CSA 的简化版）

HCA 把 m' = **128**（CSA 的 32 倍）个 token 压成 1 个 entry，**没有 indexer 也没有 top-k**——直接用所有压缩后的 entry 做 dense MQA。

公式更简洁（PDF 公式 22-23）：

```
S = Softmax(Z + B)                # 压缩权重
C^Comp_i = Σ_j S_j ⊙ C_j          # 加权压缩
```

然后直接 MQA。

**为什么不用稀疏？**—— 因为 m'=128 已经把 1M 压到 7800 多个 entry 了，稠密注意力完全扛得住。

### 5.6 三个关键的"工程细节"（论文 §2.3.3）

#### (1) 部分 RoPE（Partial Rotary Positional Embedding）

不是给整个 query/key 都加位置编码，**只给最后 64 维**加 RoPE。
而且因为压缩后 entry 既是 K 又是 V，会带上"绝对位置"，所以输出端也要给 o_t 的最后 64 维加上 **位置 -i 的 RoPE** 来抵消，让最终输出携带的是相对位置信息。

> 这是个非常巧的小 trick：用"反向 RoPE"在输出上抵消"输入中混入的绝对位置"。

#### (2) 滑窗补丁（Sliding Window Branch）

CSA / HCA 都有一个"看不到当前压缩块内 token"的死角——比如压缩比 m=4 时，第 7 个 token 看不到第 5、6 这种邻居。

解决：**额外加一条 sliding window 分支**，每个 query 直接看最近 n_win = **128** 个未压缩的 token。这条分支和压缩注意力**拼接**起来一起做 MQA。

→ 这就是为什么 V4 同时有两种 KV：**压缩 KV（CSA/HCA 的）+ 滑窗 KV（最近 128 token 的原始 KV）**。后面讲 KV cache 布局时还会回来。

#### (3) Attention Sink

经典 trick（见 OpenAI 的 GPT-OSS、StreamingLLM）。在 softmax 分母里加一个**可学习的固定项 exp(z'_h)**：

```
score = exp(logit) / (Σ exp(logits) + exp(z'_h))
```

含义：允许某些 head 的总注意力权重**不必加到 1**，甚至可以接近 0——也就是允许"这个 head 这一步啥都不看"。在长上下文里非常有用，避免 head 被强迫去关注无意义的内容。

### 5.7 效率账（这才是 V4 的杀手锏）

把 BF16 GQA8（head_dim=128）当基线（典型 LLM 配置），1M context 下 V4 的 KV cache **只有它的 ~2%**。

精度上还做了**混合存储**：
- RoPE 那 64 维 → BF16（位置敏感，精度要紧）
- 其余维度 → FP8
- Indexer 的 QK 路径 → **FP4**

最重要的：CSA 的 top-k 可以做得比 V3.2 更小（更少候选），短/中长度也变快。

---

## 6. 创新 ③：Muon 优化器

### 6.1 你已经会的：AdamW

AdamW 是目前 LLM 训练的事实标准：
- 维护每个参数的一阶动量 m 和二阶动量 v
- 用 m / √v 做归一化更新
- 加权重衰减

它是**逐元素（element-wise）**的——每个参数独立更新。

### 6.2 Muon 的核心想法：把更新"正交化"

Muon 不再逐元素，而是**矩阵级别**的优化器：

```
G_t = ∇W L                          # 梯度
M_t = μ M_{t-1} + G_t                # 累积动量
O'_t = HybridNewtonSchulz(μ M_t + G_t)  # 关键：把它"正交化"
W_t = W_{t-1} (1 - η λ) - η · O'_t · √max(n,m) · γ
```

第三行的 `HybridNewtonSchulz` 是关键：它把矩阵 M 近似成它的"正交化版本" U·V^T（其中 M = UΣV^T 是 SVD 分解）。

**为什么要正交化？**—— 直觉是：
- 普通梯度下降的方向 G 可能在某些奇异值上特别大，某些上特别小，导致更新"偏科"。
- 正交化 = 把所有奇异值压成 1 → 各个方向上都用一样的"步长"更新 → 更稳、更快。

### 6.3 V4 的"混合 Newton-Schulz 迭代"

直接做 SVD 太贵，所以用 Newton-Schulz 迭代近似。每一步是：

```
M_k = a M_{k-1} + b (M_{k-1} M_{k-1}^T) M_{k-1} + c (M_{k-1} M_{k-1}^T)^2 M_{k-1}
```

V4 用 **10 次迭代**，分两阶段：
- **前 8 次**：用激进系数 (a, b, c) = (3.4445, -4.7750, 2.0315)，**快速把奇异值拉到 1 附近**。
- **后 2 次**：用温和系数 (2, -1.5, 0.5)，**精细稳定到 1**。

> 类比：开车先猛踩油门加速到目标速度，再松油门微调，避免来回震荡。

### 6.4 哪些层用 Muon？

V4 不是**全部**用 Muon——下面这些**继续用 AdamW**：
- Embedding 层（嵌入是 lookup table，逐元素更新更合适）
- 预测头
- 所有 RMSNorm 的权重
- mHC 的静态偏置和 gating

为什么？—— 这些参数要么是 lookup（不需要矩阵正交化），要么是标量/向量（没有矩阵结构），强行 Muon 没意义还可能搞坏。

> ✅ **Muon 一句话总结**：把矩阵更新方向"正交化"，让训练在所有方向上步长均匀，从而更快更稳。

---

## 7. 工程基础设施（这部分对应你以后的并行计算课）

V4 的**算法创新**前面讲完了，但论文有整整 10 页在讲**工程**——因为算法再好，跑不出来等于零。

### 7.1 MegaMoE：通信-计算融合的 Mega-Kernel

#### 问题背景

MoE 的瓶颈在**专家并行（Expert Parallelism, EP）**：每个 token 要被送（dispatch）到分布在不同 GPU 的专家上算，算完再收（combine）回来。这两步是**机间 All-to-All 通信**，超贵。

一个 MoE 层的执行流：

```
Dispatch (通信) → Linear-1 (计算) → 激活 → Linear-2 (计算) → Combine (通信)
```

#### V4 的做法：把它们流水线化

V4 的关键洞察：**通信时间 < 计算时间**。所以只要把通信"藏到"计算后面，整体延迟就只剩计算的部分。

具体方案：把专家分成多波（waves）调度：

```
Wave 1:  Dispatch → L1 → Act → L2 → Combine
Wave 2:           Dispatch → L1 → Act → L2 → Combine
Wave 3:                    Dispatch → L1 → Act → L2 → Combine
                ↑           ↑           ↑
        三个 wave 在 GPU 上同时进行
```

每一时刻，**当前 wave 在算，上一 wave 在 combine，下一 wave 在 dispatch**——三件事一起做。

#### 收益和影响

- 比 baseline 快 **1.50× ~ 1.73×**（推理）
- RL rollout 等长尾场景快到 **1.96×**
- 开源叫 **MegaMoE**，是 DeepGEMM 的一部分

#### 一个非常硬核的"硬件设计建议"

论文给 GPU 厂商提了一个公式：

```
C / B ≤ V_comp / V_comm     (要让通信能完全藏在计算里)
```

其中 C = 算力峰值 (FLOP/s)，B = 互联带宽 (B/s)。对 V4-Pro 来说算出来是：

```
C / B ≤ 6144 FLOPs / Byte
```

**翻译**：每 1 GBps 互联带宽够"藏"6.1 TFLOP/s 计算。**带宽到这个比例就够了，再加更多带宽是浪费**——这是给 NVIDIA / 华为的设计指引。

### 7.2 TileLang：DSL 写 Kernel

V4 的算子复杂到一般写法要用几百个 PyTorch 算子，性能很差。

DeepSeek 的解法：用 **TileLang**（一个领域特定语言 DSL）写**融合 kernel**。它有几个亮点：

1. **Host Codegen**：CPU 端的"参数检查""shape 校验"全部生成成 C++ 代码，避开 Python 的开销。**单次调用开销从几十微秒降到 < 1 微秒**。
2. **集成 Z3 SMT 求解器**：编译时用形式化方法证明整数表达式的性质，自动启用更激进的优化（比如向量化、栅栏插入）。
3. **位级可复现**：默认关掉 fast-math，对齐 NVCC 的求值顺序，能做到和手写 CUDA **完全一致的 bit-by-bit 输出**。

> 对你的并行课启发：现代异构计算的趋势就是 "DSL + 编译器"，而不是手写 CUDA。

### 7.3 批不变 + 确定性内核

#### 批不变（Batch Invariance）

定义：**同一个 token 的输出，不论它在 batch 的哪个位置，结果都是 bit-by-bit 相同的**。

为什么要？—— 因为：
- 调试时方便复现 bug
- 训练 / 推理之间 bit-aligned，避免行为漂移

实现难点：
- **Attention**：传统 split-KV 把同一序列拆到多个 SM，会破坏批不变。V4 设计了**双 kernel 策略**——主 kernel 一个 SM 处理一序列保证不变性，第二 kernel 处理"末尾零碎波"减少波量化损失。
- **MatMul**：cuBLAS 不能保证批不变，V4 全部换成 **DeepGEMM**，并放弃 split-k 优化（重新写了优化版）。

#### 确定性（Determinism）

不确定性来自 atomicAdd 之类的"乱序累加"。V4 的解法：
- **Attention 反传**：每个 SM 独立累加 buffer，最后再做确定性求和。
- **MoE 反传**：用 token 顺序预处理 + buffer 隔离。
- **mHC MatMul**：split-k 的输出分别保存，后续 kernel 做确定性 reduce。

### 7.4 FP4 量化感知训练（QAT）

#### 什么是 FP4？

float4 ——只有 4 bit 来表示一个浮点数。MXFP4 用 E2M1（2 位指数 + 1 位尾数 + 1 位符号）。

#### V4 把 FP4 用在哪？

1. **MoE 专家权重**——这是 GPU 显存占用的大头。
2. **CSA 里 indexer 的 QK 路径**——长 context 下 indexer 算分是热点。
3. 顺便把 indexer 的 index_score 从 FP32 量化到 BF16。

→ Indexer 整体快 **2×**，KV recall 仍保持 99.7%。

#### 关键 trick：FP4 → FP8 的"无损反量化"

V4 的训练**仍然用 FP8 跑前向**，但权重是 FP4 存储的——前向时把 FP4 反量化回 FP8，反向时直通到 FP32 master weight（**Straight-Through Estimator, STE**）。

为什么"无损"？—— 因为 FP8 (E4M3) 比 FP4 (E2M1) 多 2 位指数（动态范围大得多）。只要 FP4 子块的 scale 比例没爆，全部信息能塞进 FP8。这样**完全复用 FP8 训练流水线**，不用改一行代码。

> 这个 trick 非常工程派——把"数学性质（FP8 动态范围 > FP4 量化的 scale 比例）"利用到极致来省工程量。

### 7.5 训练框架：mHC 的省内存实现 / Muon + ZeRO / 上下文并行

简要列出关键点（细节翻 PDF §3.5）：

**Muon 和 ZeRO 的冲突**  
ZeRO 把每个参数矩阵切到多卡上（每卡只存一段）。但 Muon 的 Newton-Schulz 需要**完整的梯度矩阵**才能正交化。  
V4 的解：用**背包算法（knapsack）**把矩阵分配到 ZeRO 节点（限制每节点最多 5 个矩阵），padding 开销 < 10%。MoE 参数则按"down → up → gate"顺序展平后均匀分布。

**两阶段上下文并行（CP）应对压缩注意力**  
传统 CP 把序列按位置切到多卡。但 V4 的压缩需要 **m 个连续 token**，可能跨越两卡边界。  
V4 的解：
1. **第一阶段**：每个 rank 把自己最后 m 个未压缩 token 发给下一 rank。
2. **第二阶段**：跨所有 rank 做 all-gather 收集压缩 KV。

**张量级激活检查点**  
传统 checkpointing 的粒度是"整层"。V4 用 **TorchFX 追踪计算图**，给 tensor 级粒度——在反向需要时再重算最小子图。这样能在内存 / 算力之间做更精细的权衡。

### 7.6 推理框架：异构 KV Cache

#### 挑战

V4 一个推理请求里同时有几种 KV：
- CSA 的压缩 KV（每 m 个 token 1 个）
- HCA 的压缩 KV（每 m' 个 token 1 个）
- Sliding Window 的未压缩 KV（最近 n_win 个）
- CSA / HCA 不足 m 个的"尾部 buffer token"
- CSA 的 indexer KV（额外维度）

这违反了 PagedAttention 的基本假设（每 block 固定 token 数）。

#### V4 的解法

把 KV 分成两块管理：

1. **State Cache**（状态缓存）  
   存 SWA KV + 未压缩的尾部 token。每个请求一个**固定大小**的 block。直接当"状态空间模型"管理。

2. **Classical KV Cache**（经典缓存）  
   存 CSA / HCA 的压缩 KV。每 block 覆盖 **lcm(m, m')** 个原始 token——**最小公倍数**！这样 CSA 和 HCA 的对齐都能满足。

#### 磁盘 KV Cache：共享前缀复用

很多对话有共享 system prompt。V4 把所有压缩 KV 写到磁盘，下次命中时直接读。

SWA KV 太大（约 8× 压缩 KV），有三种策略：
- **Full SWA Caching**：全存。零计算冗余但磁盘 I/O 不平衡。
- **Periodic Checkpointing**：每 p 个 token 存一次。可调权衡。
- **Zero SWA Caching**：完全不存，命中时基于 CSA/HCA 重算最后 n_win × L 个 token 重建。

---

## 8. 训练流水线

### 8.1 预训练数据 + 长度课程

- 数据：Flash 32T tokens，Pro 33T tokens，词表仍是 128K。
- **序列长度递进**：4K → 16K → 64K → 1M。
- **稀疏注意力切入时机**：先用 dense attention 跑前 1T tokens 热身，到 **64K 长度**才切到稀疏 + 压缩。
- 学习率：Flash 峰值 2.7e-4，Pro 峰值 2.0e-4，最后 cosine decay 到 1/10。

### 8.2 训练稳定性两板斧（这部分论文很坦率）

V4 训练时频繁出 loss spike，作者发现 spike 都伴随 **MoE 层的 outlier**。两个救命招：

#### (1) Anticipatory Routing（前瞻路由）

**问题诊断**：路由网络和 backbone 同步更新会形成"路由抖动→更多 outlier→更大抖动"的恶性循环。

**做法**：第 t 步用**当前**参数 θ_t 算 feature，但路由 index 用 **θ_{t-Δt}**（早一点的快照）算。

**怎么省开销**：在第 t-Δt 步预先 fetch 第 t 步的数据，**提前**算好路由 index 缓存起来，第 t 步直接拿来用。多花约 20% wall-time。

**进一步优化**：**自动检测 + 动态启用**——只在 loss spike 时切到这个模式，平时关掉，最终额外开销几乎可忽略。

#### (2) SwiGLU Clamping

直接给 SwiGLU 的 linear 部分 clamp 到 [-10, 10]，gate 部分 ≤ 10。粗暴但有效。

> 论文很诚实地说："我们也不完全理解为什么这两招有效，留给社区研究。"

### 8.3 后训练：Specialist + On-Policy Distillation

V4 的后训练**整体替换了 V3.2 的"混合 RL"**。流程：

```
                       Base Model
                            │
                            ▼
        ┌───────┬───────┬───────┬───────┐
        │       │       │       │       │
       SFT₁   SFT₂   SFT₃    ...    SFT_N
        │       │       │       │       │
        ▼       ▼       ▼              ▼
       RL₁    RL₂    RL₃    ...    RL_N
       (GRPO + 各领域 reward)
        │       │       │              │
        ▼       ▼       ▼              ▼
      数学    代码    Agent          指令
      专家    专家    专家            专家
        │       │       │              │
        └───────┴───┬───┴──────────────┘
                    │
                    ▼
            On-Policy Distillation
            (单一 student 学 ≥10 个 teachers)
                    │
                    ▼
              Final V4 Model
```

#### Specialist Training（专家训练）

每个领域（数学、代码、Agent、指令跟随……）单独训练一个专家：

1. SFT（监督微调）打基础。
2. **GRPO**（Group Relative Policy Optimization，DeepSeek-R1 那个算法）做 RL，每个领域用专属 reward。

每个专家**还按"思考预算"训练 3 个版本**（Non-think / Think High / Think Max），靠不同的长度惩罚和 context 大小区分。

#### Generative Reward Model（GRM）

难验证的任务（写作、推理）传统上要训练一个 scalar reward model。V4 不要——**让 actor 网络自己当 reward 评估者**，用 rubric 引导 + RL 优化 GRM 本身。

直觉：让模型既"会写"也"会评"，两件事一起优化时它的评判会受自己的推理能力支持，更鲁棒。

#### On-Policy Distillation（OPD）

最后把 N 个专家融合成 1 个 student，目标函数是：

```
L_OPD(θ) = Σ_i w_i · KL( π_θ ∥ π_E_i )
```

**关键不同点**：
- **反向 KL**（学生比教师，让学生**专注于教师的高概率区域**），不是普通蒸馏的正向 KL。
- **on-policy**：训练数据由**学生自己采样**，不是教师采样——保证学生学到的是它实际会遇到的分布。
- **全词表 logits 蒸馏**（不是只对当前 token 算 KL）——更稳定、更忠实。

#### 全词表 OPD 的工程难点

|V| > 100k，N ≥ 10 个 teacher，每步要算 N 套 logits——不可能全存内存。  
解法：
- 教师权重存中央存储，按需加载（ZeRO-like sharding）。
- 只缓存教师**最后一层 hidden state**，到训练时再过 prediction head 现算 logits。
- 按教师 index 排序训练样本，**每个 mini-batch 同一 teacher head 只加载一次**。
- KL 的精确计算用专门的 TileLang kernel。

### 8.4 其他后训练亮点

- **Tool-call schema** 用 `|DSML|` token + XML 格式（比 JSON 少转义错误）。
- **Interleaved Thinking**：Tool-calling 场景**保留所有历史 reasoning**（之前 V3.2 是丢弃的），让 agent 长任务里思路连贯。
- **Quick Instruction**：在输入里追加一组特殊 token（`<|action|>`, `<|query|>`, `<|domain|>`...）让某些"前置任务"（要不要搜、什么领域）**复用 KV cache 并行算**，省一个独立小模型。
- **DSec 沙箱**（Rust 写的 Apiserver / Edge / Watcher 三件套）支持 Function Call / Container / microVM / fullVM 四种隔离级别，单集群上百万实例。

---

## 9. 关键参数速查表

```
┌──────────────────┬──────────────────┬──────────────────┐
│                  │ V4-Flash         │ V4-Pro           │
├──────────────────┼──────────────────┼──────────────────┤
│ 总参数            │ 284B             │ 1.6T             │
│ 激活参数          │ 13B              │ 49B              │
│ Transformer 层数  │ 43               │ 61               │
│ 隐藏维度 d        │ 4096             │ 7168             │
│ 路由专家          │ 256              │ 384              │
│ 共享专家          │ 1                │ 1                │
│ 激活专家 / token  │ 6                │ 6                │
│ 专家中间维度      │ 2048             │ 3072             │
│ Attention 头数 n_h│ 64               │ 128              │
│ 头维度 c          │ 512              │ 512              │
│ Q 压缩维度 d_c    │ 1024             │ 1536             │
│ CSA 压缩比 m      │ 4                │ 4                │
│ HCA 压缩比 m'     │ 128              │ 128              │
│ CSA top-k         │ 512              │ 1024             │
│ 滑窗 n_win        │ 128              │ 128              │
│ Indexer 头数      │ 64               │ 64               │
│ Indexer 头维度    │ 128              │ 128              │
│ 输出投影组数 g    │ 8                │ 16               │
│ mHC n_hc          │ 4                │ 4                │
│ Sinkhorn 迭代数   │ 20               │ 20               │
│ MTP 深度          │ 1                │ 1                │
│ 预训练 tokens     │ 32T              │ 33T              │
│ 峰值 LR           │ 2.7e-4           │ 2.0e-4           │
│ 最大 batch (tok)  │ 75.5M            │ 94.4M            │
└──────────────────┴──────────────────┴──────────────────┘
```

---

## 10. V4 vs V3.2：差量对比（一图速记）

```
                  V3.2                          V4
                   │                             │
┌─ Attention ──────┼─────────────────────────────┼─────────────────────┐
│                  MLA + DSA                     CSA + HCA  (混合)     │
│                  (V3.2 已经是 sparse)          (再加 token 压缩)     │
│                                                                     │
├─ Residual ───────┼─────────────────────────────┼─────────────────────┤
│                  普通残差                      mHC (双随机矩阵约束)   │
│                                                                     │
├─ Optimizer ──────┼─────────────────────────────┼─────────────────────┤
│                  AdamW (全部)                  Muon + AdamW (混合)   │
│                                                Newton-Schulz 正交化  │
│                                                                     │
├─ Quantization ───┼─────────────────────────────┼─────────────────────┤
│                  FP8                           FP8 + FP4 QAT         │
│                                                                     │
├─ Post-training ──┼─────────────────────────────┼─────────────────────┤
│                  SFT + 混合 RL                 Specialist + OPD       │
│                                                                     │
├─ Routing ────────┼─────────────────────────────┼─────────────────────┤
│                  Sigmoid 打分                  Sqrt(Softplus) 打分   │
│                  全部学习路由                  前 3 层 Hash 路由     │
│                                                                     │
├─ Stability ──────┼─────────────────────────────┼─────────────────────┤
│                  常规                          Anticipatory Routing  │
│                                                + SwiGLU Clamp        │
│                                                                     │
├─ Engineering ────┼─────────────────────────────┼─────────────────────┤
│                  DeepGEMM                      MegaMoE + TileLang    │
│                                                批不变 + 确定性       │
│                                                异构 KV cache + 落盘  │
│                                                                     │
└─ Context ────────┼─────────────────────────────┼─────────────────────┘
                  ~128K → 笨拙                   1M → 原生高效
                                                 (FLOPs 27%, KV 10%)
```

---

## 11. 学习路径建议

如果你想真正掌握这些内容，建议按下面的顺序：

### 第 1 周：把"地基"打牢
- 重新读你笔记里的 `flash_attn_1`、`mha_gqa_mqa_mla_dsa/attention_variants.md`、`linear_attn_mamba_gdn`。
- 重点搞懂：标准注意力为什么是 O(n²)？KV cache 是什么？MLA 怎么压缩的？DSA 怎么做 top-k？
- ✅ 读完应该能回答："V3.2 已经做了啥，还差什么。"

### 第 2 周：V4 注意力（本文 §5）
- 读论文 §2.3 的 CSA / HCA 公式 + 图 3、图 4。
- 自己用纸笔推一遍：n=16, m=4 时 CSA 怎么把 16 个 token 压成 4 个 entry？再用 top-k=2 选哪 2 个？
- 配套：去看 [HuggingFace 的 DeepSeek-V4-Pro 代码](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference)，对照公式看实现。

### 第 3 周：mHC + Muon（本文 §4、§6）
- 学一点矩阵论：双随机矩阵、谱范数、SVD。
- 实现一个 toy 版的 Sinkhorn-Knopp（不到 10 行 Python）。
- 跑一遍 Newton-Schulz 迭代，验证它确实把奇异值压到 1。

### 第 4 周：工程基础设施（本文 §7）
- 这部分对应你将来的并行计算课。
- 重点理解：MoE 的 EP 通信瓶颈、计算/通信重叠、批不变性、为什么需要确定性训练。
- 推荐配套读：[MegaMoE PR](https://github.com/deepseek-ai/DeepGEMM/pull/304)，看一眼真实的 mega-kernel 长什么样。

### 第 5 周：训练 / 后训练（本文 §8）
- 学一点 RL 基础：GRPO、PPO、reward model。
- 重点理解：为什么 OPD 用反向 KL？为什么是 on-policy？为什么要全词表？

---

## 12. 总结：用一张图记住 V4

```
        V4 = "让百万 token 上下文变便宜"
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     算法创新       工程优化       训练方法
        │              │              │
   ┌────┴────┐   ┌─────┴─────┐    ┌────┴────┐
   │         │   │           │    │         │
 mHC   CSA + HCA MegaMoE  TileLang 专家训练 + OPD
   │       │       │         │       │
 残差    长上下文  通信计算  DSL+SMT  反向 KL +
 守恒    主武器    重叠     可复现   全词表蒸馏
                                     │
                              + Anticipatory Routing
                              + SwiGLU Clamp（稳训练）
                              + GRPO（专家 RL）
                              + GRM（生成式奖励）
```

---

## 13. 参考资料

主要资料：
- 📄 [DeepSeek-V4 Technical Report (PDF)](./DeepSeek_V4.pdf) —— 本笔记的唯一权威来源
- 🤗 [DeepSeek-V4-Pro on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- 🤗 [DeepSeek-V4-Flash on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- 🛠 [MegaMoE / DeepGEMM PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304)

社区解读（用于交叉验证）：
- [DeepSeek-V4 Review on Medium (Andrew Lukyanenko)](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1)
- [NVIDIA Blog: Build with DeepSeek V4](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/)

本仓库相关前置笔记：
- [Flash Attention 1](../flash_attn_1/flash_attention_1.md)
- [MHA → MQA → GQA → MLA → DSA](../mha_gqa_mqa_mla_dsa/attention_variants.md)
- [Linear Attention / Mamba / GDN](../linear_attn_mamba_gdn/)

---

> 📝 这份笔记**不会代替你读论文**，但它能让你读论文时**有一张地图**——知道每个公式属于哪个大模块，以及它要解决什么问题。
>
> 真正读懂 V4 没有捷径。但每搞懂一个模块（比如 CSA 的压缩怎么做），你就能在面试、读其他论文、搞自己实验时**直接复用**。这才是这份学习的真正价值。
