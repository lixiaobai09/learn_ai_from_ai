# Linear Attention、Mamba 与 Gated DeltaNet：从注意力机制到现代线性序列模型

> 面向有一定 CS 背景、刚接触深度学习/Transformer 的同学。本讲义按"为什么需要它 → 它怎么做 → 它的局限"展开，每一节都先给直觉，再给公式。

---

## 0. 总览：一句话先记住

**Linear Attention、Mamba、Gated DeltaNet 都是"隐状态是一个矩阵的 RNN"。**

它们的共同形式可以写成：

$$
S_t = G_t \cdot S_{t-1} + u_t v_t^\top, \qquad y_t = S_t^\top q_t
$$

- $S_t \in \mathbb{R}^{d\times d}$：状态矩阵（可以理解成一个"键到值的字典"）
- $G_t$：状态如何衰减/转移的算子
- $u_t v_t^\top$：本步要写入的新内容
- $q_t$：本步要读取时用的查询

**三者的区别只在 $G_t$ 是什么样：**

| 模型 | $G_t$ 形式 | 直觉 |
|---|---|---|
| Linear Attention | $G_t = I$（恒等） | 黑板一直叠加 |
| Mamba (S6) | $G_t = \mathrm{diag}(\alpha_t)$（输入相关的对角矩阵） | 信号滤波器，可选择性遗忘 |
| Gated DeltaNet | $G_t = \alpha_t (I - \beta_t k_t k_t^\top)$ | 带全局衰减 + 局部覆盖的 KV 字典 |

记住这张表。后面所有内容都是对它的展开。

---

## 1. 出发点：为什么标准 Attention 不够用

标准 Transformer 的注意力公式：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

设序列长度为 $N$，特征维度为 $d$。

### 1.1 时间复杂度：$O(N^2 d)$

注意力矩阵 $A = QK^\top$ 是一个 $N\times N$ 矩阵，单是构造它就需要 $O(N^2 d)$ 次乘加。再乘 $V$ 又是 $O(N^2 d)$。当 $N=128\text{k}$ 时，$N^2 = 1.6\times 10^{10}$，光这个矩阵就放不进显存。

### 1.2 推理时的 KV cache：$O(N d)$ 显存

自回归生成时，每生成一个新 token 都要看见所有历史的 $K, V$，因此必须把它们缓存下来。生成第 $N$ 个 token 时，KV cache 已经是 $O(N d)$ 显存。这是为什么 long-context LLM 推理那么贵。

### 1.3 关键瓶颈在哪儿

注意 softmax：$\mathrm{softmax}(QK^\top)$ 让 $Q$ 和 $K$ "锁死"在一起——你必须先算出整个 $QK^\top$，才能做 softmax。这就破坏了一个潜在的优化机会：**矩阵结合律**。

如果没有 softmax，我们本可以这样算：

$$
(QK^\top)V = Q(K^\top V)
$$

后者只需要先算 $K^\top V$（$d\times d$ 矩阵），复杂度 $O(Nd^2)$；再乘 $Q$，又是 $O(Nd^2)$。**总共 $O(Nd^2)$，对 $N$ 是线性的！**

下面三种方法都在变着法子绕开 softmax，享受这个线性。

---

## 2. Linear Attention（2020）：最朴素的线性化

### 2.1 核心想法：用核函数代替 softmax

把 softmax 换成一个非负的特征映射 $\phi:\mathbb{R}^d\to\mathbb{R}^{d'}$（比如 $\phi(x)=\mathrm{elu}(x)+1$ 或 $\phi(x)=\mathrm{ReLU}(x)$）：

> **补充：ReLU 和 ELU 的函数原型**
>
> ReLU（Rectified Linear Unit，整流线性单元）：
>
> $$
> \mathrm{ReLU}(x) = \max(0, x) = \begin{cases} x, & x \ge 0 \\ 0, & x < 0 \end{cases}
> $$
>
> 正半轴恒等、负半轴归零。导数 $\mathrm{ReLU}'(x)=1\ (x>0)$ 或 $0\ (x<0)$。优点是计算极快、缓解梯度消失；缺点是负半轴梯度恒为 0，可能导致 "dying ReLU"（神经元一旦进入负区就再也学不到东西）。
>
> ELU（Exponential Linear Unit，指数线性单元）：
>
> $$
> \mathrm{ELU}(x) = \begin{cases} x, & x \ge 0 \\ \alpha(e^x - 1), & x < 0 \end{cases} \qquad (\alpha > 0,\ \text{通常取 } 1)
> $$
>
> 正半轴和 ReLU 相同；负半轴是平滑的指数曲线，从 0 渐近到 $-\alpha$。优点是处处光滑、不会"死亡"、输出均值更接近 0，训练更稳；缺点是要算 $e^x$，比 ReLU 慢。
>
> 在线性注意力里用的是 $\phi(x) = \mathrm{ELU}(x) + 1$，正好是恒正函数：$x \ge 0$ 时取 $x+1$，$x<0$ 时取 $e^x$（值域 $(0, 1)$）——既满足"非负"要求，又处处光滑可导。

把 softmax 换成 $\phi$ 后，第 $t$ 个 token 的输出可以写成：

$$
y_t = \frac{\sum_{i\le t} \phi(q_t)^\top \phi(k_i)\, v_i}{\sum_{i\le t}\phi(q_t)^\top \phi(k_i)}
$$

这一步是怎么来的？下面分五步推导，让你看清每一处的逻辑。

#### 推导：从标准 Attention 一步步到这个公式

**Step 1：写出标准 Attention 的"逐 token"形式**

教科书上 Attention 通常写成矩阵形式 $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt d)V$。但我们关心**第 $t$ 个 query 对应的输出 $y_t$ 长什么样**，把它展开：

$$
y_t = \sum_{i=1}^{N} \underbrace{\frac{\exp(q_t^\top k_i / \sqrt{d})}{\sum_{j=1}^{N}\exp(q_t^\top k_j / \sqrt{d})}}_{\text{第 } i \text{ 个 token 的注意力权重}} \cdot v_i
$$

读法：分子 $\exp(q_t^\top k_i)$ 度量 $q_t$ 和 $k_i$ 的"匹配程度"；分母对所有 $j$ 归一化让权重和为 1。下面省略 $\sqrt d$（缩放因子，不影响推导结构）：

$$
y_t = \sum_{i=1}^{N} \frac{\exp(q_t^\top k_i)}{\sum_j \exp(q_t^\top k_j)} v_i
$$

**Step 2：把 $\exp(q^\top k)$ 看成一个"相似度核函数"**

注意 $\exp(q_t^\top k_i)$ 干的事就是"输入两个向量、输出一个非负实数"。这种形式的函数叫**核函数**（kernel function）。给它一个抽象名字：

$$
\kappa(q, k) := \exp(q^\top k)
$$

那么 Attention 公式重写成：

$$
y_t = \sum_{i} \frac{\kappa(q_t, k_i)}{\sum_j \kappa(q_t, k_j)} v_i
$$

这一步**只是换符号，没改任何东西**。

**Step 3：核函数的"分解"——Mercer 定理的直觉**

数学上有一个深刻的结论（**Mercer 定理**，SVM 课会讲）：

> 一个"足够好"的核函数 $\kappa(q, k)$，总可以分解成两个特征映射的内积：
> $$\kappa(q, k) = \phi(q)^\top \phi(k)$$
> 其中 $\phi(\cdot)$ 是一个把向量映射到（可能更高维）特征空间的函数。

直观例子：
- $\kappa(q,k)=q^\top k$ → $\phi(x)=x$（恒等映射）
- $\kappa(q,k)=(q^\top k)^2$ → $\phi(x)$ 是 "$x$ 的所有二次单项式组成的向量"
- $\kappa(q,k)=\exp(q^\top k)$ → 也能分解，但 $\phi$ 是**无穷维的**（用泰勒展开能看出来）

**问题**：原版 attention 的 $\kappa=\exp(\cdot)$ 对应无穷维的 $\phi$，**没法在计算机里直接表示**。

Linear Attention 的做法是**反过来**：不去找 $\exp$ 的分解，而是**直接挑一个简单的、有限维的 $\phi$**（比如 $\phi(x)=\mathrm{ELU}(x)+1$），然后**定义新的核**：

$$
\kappa_\phi(q, k) := \phi(q)^\top \phi(k)
$$

这是**关键的妥协**：我们牺牲了 softmax 的精确表达力，换来 $\phi$ 是有限维、可计算的。代入 Step 2 的公式：

$$
y_t = \sum_{i} \frac{\phi(q_t)^\top \phi(k_i)}{\sum_j \phi(q_t)^\top \phi(k_j)} v_i \qquad (\star)
$$

**Step 4：加上因果性（Causal Mask）**

LLM 是自回归的，预测第 $t$ 个 token 时只能看见 $i\le t$ 的部分。所以求和范围限制为 $i\le t$（分母里的 $j$ 同理）：

$$
y_t = \sum_{i \le t} \frac{\phi(q_t)^\top \phi(k_i)}{\sum_{j \le t} \phi(q_t)^\top \phi(k_j)} v_i
$$

**Step 5：把分母提到求和外面**

分母里的求和变量是 $j$，**和外层 $\sum_i$ 的 $i$ 完全无关**——所以分母对外层求和来说就是一个常数，可以提到 $\sum_i$ 外面：

$$
y_t = \frac{1}{\sum_{j \le t} \phi(q_t)^\top \phi(k_j)} \cdot \sum_{i \le t} \phi(q_t)^\top \phi(k_i)\, v_i
$$

写成一个分式（分母里的 $j$ 是 dummy 变量，可以换回 $i$）：

$$
\boxed{\;y_t = \frac{\sum_{i\le t} \phi(q_t)^\top \phi(k_i)\, v_i}{\sum_{i\le t}\phi(q_t)^\top \phi(k_i)}\;}
$$

#### 这一步真正"做了什么"？

| 步骤 | 形式 | 改变 |
|---|---|---|
| Step 1 | $\sum_i \frac{\exp(q^\top k_i)}{\sum_j \exp(q^\top k_j)} v_i$ | 标准 attention 展开 |
| Step 2 | 把 $\exp$ 抽象成 $\kappa(\cdot,\cdot)$ | 只是换符号 |
| **Step 3** | **把 $\kappa$ 替换为 $\phi(q)^\top\phi(k)$** | **核心妥协：放弃 softmax** |
| Step 4 | 加 causal mask $i\le t$ | 适配自回归 |
| Step 5 | 提分母，整理 | 纯代数变形 |

**最关键的就是 Step 3**——其余几步都是符号操作，只有 Step 3 是真正的"理论假设"：用一个能拆成内积的核 $\phi(q)^\top\phi(k)$ 替代了 $\exp(q^\top k)$。这个替代换来了下面巨大的好处。

#### 为什么这一步换得值？

注意分子里 $\phi(q_t)^\top \phi(k_i)$ 是个标量，和 $v_i$ 相乘后再求和。利用矩阵结合律：

$$
\sum_{i\le t}\phi(q_t)^\top\phi(k_i) v_i = \phi(q_t)^\top \underbrace{\sum_{i\le t}\phi(k_i) v_i^\top}_{S_t}
$$

于是得到一个非常干净的递推：

$$
\boxed{\;S_t = S_{t-1} + \phi(k_t)\, v_t^\top, \qquad y_t = \phi(q_t)^\top S_t\;}
$$

（分母同理，用一个向量 $z_t$ 跟踪。）

### 2.2 这其实是一个矩阵状态的 RNN

把 $S_t$ 看作"隐状态"，它的维度是 $d'\times d$，**是一个矩阵**——这是它和传统 RNN（隐状态是向量）最大的差别。

直觉：$S_t$ 像一块黑板，每来一个 token，就把 $\phi(k_t) v_t^\top$ 这块"补丁"叠加上去。读取时用 $\phi(q_t)$ 当索引去读。

### 2.3 训练时怎么并行？

虽然递推看起来是串行的，但因为只是简单的累加，训练时可以用 **prefix sum / cumsum** 完全并行：所有时间步的 $S_t$ 可以在 $O(\log N)$ 深度内一次性算出。这就是 Linear Attention 既能像 RNN 一样推理、又能像 Transformer 一样训练的原因。

#### "$O(\log N)$ 深度"是什么意思？

并行计算里有两个不同的成本指标：

- **总工作量 (Work)**：所有处理器加起来要做的总操作数。
- **深度 (Depth / Span)**：假设有**无限多个处理器**，从开始到结束的"最长依赖链"长度。

举例：8 个数求和。
- 工作量永远是 7 次加法。
- 串行做：$a_1+a_2 \to (\cdots)+a_3 \to \cdots$，深度 = 7。
- 树形并行：先两两加（4 次互不依赖，可同时做）→ 再两两 → 最后一次。深度 = 3 = $\log_2 8$。

GPU 上每一"步"对应一次 kernel 同步，**深度小 ≈ 总耗时短**。

#### 为什么前缀和能做到 $O(\log N)$ 深度？

要算的是 $S_t = S_{t-1} + \phi(k_t) v_t^\top$。表面看 $S_2$ 依赖 $S_1$、$S_3$ 依赖 $S_2$……像 RNN 必须串行。

**但加法是结合律的**——这意味着可以"分块预算"。

最小例子：$a_1, a_2, a_3, a_4$，要算所有前缀和。$S_4$ 表面要等 $S_3$ 算完，但你可以**先并行算 $a_3+a_4$ 这块局部和**（不需要 $S_2$），然后只用一步 $S_4 = S_2 + (a_3+a_4)$。

#### 具体算法：Hillis-Steele Scan

对 $N=8$ 的数组 $[a_1, \dots, a_8]$，用"距离逐步翻倍"的方式同时算前缀和：

**Step 0**（初始）：
```
位置:   1    2    3    4    5    6    7    8
值:    a1   a2   a3   a4   a5   a6   a7   a8
```

**Step 1**（offset = 1）：每个位置 $i$ 同时加上位置 $i-1$ 的值
```
1: a1
2: a1+a2
3: a2+a3
4: a3+a4
5: a4+a5
6: a5+a6
7: a6+a7
8: a7+a8
```
这 7 个加法**互不依赖**，GPU 一个 kernel 同时跑完。

**Step 2**（offset = 2）：每个位置 $i$ 加上位置 $i-2$ 的值
```
1: a1
2: a1+a2
3: a1+a2+a3
4: a1+a2+a3+a4
5: a2+a3+a4+a5
6: a3+a4+a5+a6
7: a4+a5+a6+a7
8: a5+a6+a7+a8
```

**Step 3**（offset = 4）：每个位置 $i$ 加上位置 $i-4$ 的值
```
1: a1               = S_1
2: a1+a2            = S_2
3: a1+a2+a3         = S_3
4: a1+a2+a3+a4      = S_4
5: a1+a2+a3+a4+a5   = S_5
6: a1+...+a6        = S_6
7: a1+...+a7        = S_7
8: a1+...+a8        = S_8
```

**3 步搞定，所有 $S_t$ 同时出炉**，正好 $\log_2 8 = 3$ 步。

直觉：每一轮"已知的连续区间长度"翻倍——Step 1 后每个位置知道长度 ≤ 2 的前缀和，Step 2 后 ≤ 4，Step 3 后 ≤ 8……$\log N$ 轮后覆盖全长。

#### 工程上更高效的版本：Blelloch Scan

Hillis-Steele 简单但**总工作量是 $O(N\log N)$**（每轮都做 $N$ 次加法）。

更聪明的 **Blelloch scan** 用"上扫 + 下扫"两阶段（像归并树）：
- **上扫 (reduce)**：自底向上 build 一棵求和树
- **下扫 (downsweep)**：从根往下传递每个节点的前缀和

总工作量降到 $O(N)$，深度仍是 $O(\log N)$。PyTorch 的 `cumsum`、CUDA 的 `thrust::inclusive_scan`、Mamba 的 selective scan kernel，**底层都用这类算法**。

#### 回到 Linear Attention

$S_t$ 是 $d'\times d$ 矩阵，但矩阵加法照样满足结合律，parallel scan 直接适用：

- **训练时**：整个序列已知，喂给 parallel scan，$O(\log N)$ 深度并行算出所有 $S_t$，再用 $y_t = \phi(q_t)^\top S_t$ 算输出。
- **推理时**（自回归）：每次只新增一个 token，复用 $S_{t-1}$，$O(1)$ 即可。

> **核心结论**：只要操作满足**结合律**，"看似必须串行的递推"就能用 parallel scan 在 $O(\log N)$ 深度内并行算完。Mamba 的 selective scan 用的是同样思想——只是状态转移更复杂（矩阵乘法而非加法），但结合律依然成立。这是整个线性序列模型家族能在 GPU 上跑得飞快的根本原因。

### 2.4 致命短板：Key Collision（钥匙碰撞）

线性注意力是**纯加法**：$S_t = S_0 + \sum_i \phi(k_i) v_i^\top$。

设想模型先存了 $(k_1, v_1)$，过了几百步又来一个相似的 $k_{500} \approx k_1$，但要求关联到不同的 $v_{500}$。由于 $S$ 只能加不能改，旧的 $v_1$ 和新的 $v_{500}$ 会"叠加"在同一处，读出来变成两者的混合，**精确召回就崩了**。

#### 先理清"存"与"读"的机制

更新公式 $S_t = S_{t-1} + \phi(k_t) v_t^\top$ 在做的事：把 $\phi(k_t) v_t^\top$ 这块 $d'\times d$ 的"补丁"叠加到黑板 $S$ 上——可以理解为"在由 $k_t$ 决定的方向上，写下 $v_t$"。

读取时：

$$
y = \phi(q)^\top S = \sum_i \bigl(\phi(q)^\top \phi(k_i)\bigr) v_i^\top
$$

理想情况：当 $q\approx k_j$ 时，$\phi(q)^\top \phi(k_j)$ 远大于其他项，输出就是 $v_j$。这就是把状态矩阵当成"软 KV 字典"用的方式。

#### 一个手算例子：碰撞是怎么发生的

设 $d=2$，并且为简化省略 $\phi$（即 $\phi(x)=x$，不影响要展示的本质问题）。

**场景**：
- 第 1 步：存 $(k_1, v_1)$，$k_1 = [1, 0]^\top$，$v_1 = [1, 0]^\top$（含义："苹果"）
- 第 500 步：存 $(k_{500}, v_{500})$，$k_{500} = [1, 0.1]^\top$（和 $k_1$ 极相似），$v_{500} = [0, 1]^\top$（含义："香蕉"）

我们希望：查 $k_1$ 取出苹果，查 $k_{500}$ 取出香蕉。

**只看这两 token 对 $S$ 的贡献**：

$$
k_1 v_1^\top = \begin{bmatrix}1\\0\end{bmatrix}\begin{bmatrix}1 & 0\end{bmatrix} = \begin{bmatrix}1 & 0\\ 0 & 0\end{bmatrix},\quad
k_{500} v_{500}^\top = \begin{bmatrix}1\\0.1\end{bmatrix}\begin{bmatrix}0 & 1\end{bmatrix} = \begin{bmatrix}0 & 1\\ 0 & 0.1\end{bmatrix}
$$

$$
\Delta S = k_1 v_1^\top + k_{500} v_{500}^\top = \begin{bmatrix}1 & 1\\ 0 & 0.1\end{bmatrix}
$$

**用 $q = k_1$ 查询（"想要苹果"）**：

$$
y = k_1^\top \Delta S = [1, 0] \begin{bmatrix}1 & 1\\ 0 & 0.1\end{bmatrix} = [1, 1]
$$

期望 $v_1=[1,0]$，实际 $[1,1]$ —— **苹果和香蕉的混合**。

**用 $q = k_{500}$ 查询（"想要香蕉"）**：

$$
y = k_{500}^\top \Delta S = [1, 0.1] \begin{bmatrix}1 & 1\\ 0 & 0.1\end{bmatrix} = [1, 1.01]
$$

期望 $v_{500}=[0,1]$，实际 $[1,1.01]$ —— 同样是混合。**不管用哪个键查，都拿不到一个干净的值。**

#### 根本原因：两次写入在 $S$ 的同一区域不可逆地融合

两个 $(k, v)$ 是不同时间步进入状态的，本来"应该"是两条独立的记录。但因为：

1. **状态只能"加"不能"覆盖"**：第 500 步只能在 $S$ 上叠加新的 $\phi(k_{500})v_{500}^\top$，**没办法擦掉**第 1 步留下的 $\phi(k_1)v_1^\top$。
2. **键又非常相似**（$k_1 \approx k_{500}$）：两个补丁会写到 $S$ 的几乎同一区域上。

两个补丁被矩阵加法粘到一起，"$v_1$" 和 "$v_{500}$" 这两条不同信息**塌陷成了一个混合体**，再也分不开。

#### 对比 Full Attention 为什么没有这个问题

Full attention 的输出：

$$
y_t = \sum_i \frac{\exp(q_t^\top k_i)}{\sum_j \exp(q_t^\top k_j)} v_i
$$

它**不把 $(k_i, v_i)$ 压缩到固定大小的状态里**——所有 $k_i, v_i$ 原原本本保留在 KV cache 里，查询时**逐个比对**。

| | Linear Attention | Full Attention |
|---|---|---|
| 状态容量 | 固定 $d'\times d$ 矩阵 | $N$ 个 $(k,v)$ 一个不漏 |
| 写入方式 | 矩阵加法（不可逆） | 直接 append |
| 区分相似键的能力 | $\phi(q)^\top \phi(k_i)$ 是软相似度，无锐化 | softmax 用 $\exp$ 强烈放大差异，能"锁定"最匹配的那个 |

只要 $q^\top k_1$ 比 $q^\top k_{500}$ 哪怕大一点点（比如 2 vs 1.5），$\exp$ 一放大、softmax 一归一化，第一个就能拿到 0.62、第二个 0.38——**仍然能区分**。而 Linear Attention 一旦把信息揉进 $S$ 就永远揉在一起了。

#### 这促成了下一代设计

**这正是后续 DeltaNet 引入"覆盖式更新"的动机**——它用 $S(I - \beta_t k_t k_t^\top)$ 这一项**先擦掉旧的**和当前 $k$ 相关的内容，再写入新的 $v$，从根本上避免了塌陷。详见 §4.1。

实际效果：在 in-context retrieval / "needle in a haystack"任务上，纯线性注意力远远落后于 full attention。这促使后来的研究去找更好的状态更新规则。

---

## 3. Mamba（2023）：从控制论借来的状态空间模型

### 3.1 思想来源：连续状态空间方程

控制系统里描述一个动态系统常用：

$$
h'(t) = A\, h(t) + B\, x(t), \qquad y(t) = C\, h(t)
$$

- $h(t)\in\mathbb{R}^N$：系统的"内部状态"（比如电路里电容的电压）
- $x(t)$：输入信号
- $y(t)$：输出
- $A, B, C$：决定系统动力学的矩阵

这个方程的解你可能在信号与系统课见过：$h(t) = e^{At}h(0)+\int_0^t e^{A(t-s)}B x(s)\,ds$。

### 3.2 离散化：变成 RNN

时间离散化（Zero-Order Hold 或欧拉法）后：

$$
h_t = \bar A\, h_{t-1} + \bar B\, x_t, \qquad y_t = C\, h_t
$$

其中 $\bar A = e^{A\Delta}$，$\bar B$ 是相应的离散化版本，$\Delta$ 是步长。

**这就是 RNN 的形式！** 但和 LSTM/GRU 不同，它的转移矩阵来自一个连续动力系统，有更好的数学结构。

### 3.3 S4：HiPPO 与结构化矩阵

如果 $A$ 随便选，长程依赖会指数衰减或爆炸。**HiPPO 理论**给出了一个特别的 $A$ 矩阵，使得 $h(t)$ 是历史 $x(\cdot)$ 在 Legendre 多项式基下的最优投影——也就是说，状态 $h$ 本身就是历史的"压缩签名"。

S4（Structured State Space）就是用 HiPPO 初始化、并把 $A$ 限制在结构化（如对角 + 低秩）形式以加速计算。

### 3.4 S6（Mamba 的核心创新）：Selective SSM

S4 有一个大问题：$A, B, C$ 是**固定参数**，不依赖输入。这相当于"无差别记忆"——它无法根据当前看到的内容决定要记什么、忘什么。

Mamba 的关键修改是：**让 $B, C, \Delta$ 都成为输入 $x_t$ 的函数**：

$$
B_t = \mathrm{Linear}_B(x_t),\quad C_t = \mathrm{Linear}_C(x_t),\quad \Delta_t = \mathrm{softplus}(\mathrm{Linear}_\Delta(x_t))
$$

然后 $\bar A_t = \exp(\Delta_t A)$、$\bar B_t = \Delta_t B_t$（简化版）。

这样递推变成：

$$
h_t = \bar A_t h_{t-1} + \bar B_t x_t
$$

直觉：模型看到一个无信息的 token（比如停用词）时可以让 $\Delta_t \to 0$，也就是 $\bar A_t \to I, \bar B_t \to 0$，相当于"跳过"这个 token；看到一个关键 token 时让 $\Delta_t$ 大一些，把它写入状态。

**这就是"selective"**——选择性地处理输入。是 Mamba 相对 S4 最重要的突破。

### 3.5 工程关键：硬件感知的并行扫描

Selective SSM 的代价是失去了卷积形式（因为 $\bar A_t$ 是输入相关的），变回了串行 RNN。怎么训练？

Mamba 用 **Parallel Scan（并行前缀扫描）算法**（Blelloch scan）把这种依赖输入的递推也变成 $O(\log N)$ 深度的并行计算。再配合 kernel fusion，把中间状态留在 SRAM 里不写回 HBM，得到了和 Flash-Attention 同级别的硬件效率。

最终：**训练时并行（接近 Transformer），推理时 $O(1)$ 显存（像 RNN）。**

### 3.6 弱项：retrieval 不够强

Mamba 的状态 $h_t$ 是一个固定大小的向量（或多通道向量），并没有"key-value 字典"的概念。当任务需要从 prompt 里精确取出某段内容（比如"prompt 里说 user_id 是 42，请重复一遍"），Mamba 需要把这个事实压进固定大小的 $h$ 里，长 prompt 下容易丢。

实证上：Mamba 在大多数语言建模 benchmark 上很强，但在 associative recall / multi-query retrieval 类任务上明显弱于 attention。

---

## 4. Gated DeltaNet（2024）：在线学习的关联记忆

Gated DeltaNet 是 **DeltaNet** 加上 **门控**。先讲 DeltaNet。

### 4.1 DeltaNet 的关键洞见：把状态当作要"在线学习"的字典

回到 Linear Attention：$S_t = S_{t-1} + v_t k_t^\top$，纯加法。问题是 key collision。

能不能在写入 $(k_t, v_t)$ 之前，**先擦掉旧的与 $k_t$ 相关的内容**？这就是 **delta rule**：

$$
S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t)\, k_t^\top
$$

展开整理：

$$
\boxed{\;S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top\;}
$$

### 4.2 为什么这等价于"在线梯度下降"

把状态 $S$ 想成一个学习到的"键到值的线性映射"：理想情况下 $S k = v$。

定义损失（每一步的）：

$$
\mathcal{L}_t(S) = \tfrac{1}{2}\|S k_t - v_t\|^2
$$

对 $S$ 求梯度：

$$
\nabla_S \mathcal{L}_t = (S k_t - v_t)\, k_t^\top
$$

对 $S_{t-1}$ 做一步梯度下降，步长 $\beta_t$：

$$
S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top
$$

**这正是 delta rule！** 也就是说，DeltaNet 在每个时间步都对状态做一步在线梯度下降，把当前 $(k_t, v_t)$ 这对样本"学进去"。它和 60 年代的 Widrow-Hoff（LMS）算法、感知机的更新规则一脉相承。

直觉：当模型"查询" $k_t$ 时，旧状态会给一个错误答案 $S_{t-1} k_t$；delta rule 按误差大小修正，使得查 $k_t$ 时得到正确的 $v_t$。**这是"覆盖式"写入**，自然解决了 key collision。

### 4.3 加门控：可遗忘的 DeltaNet

DeltaNet 没有全局遗忘——很久之前学的东西如果一直没被覆盖，会一直留在状态里。但人类对话里经常有"换话题"的需要：早期上下文应该被"软清空"。

借鉴 Mamba 的 selective 思想，引入 **forget gate** $\alpha_t \in (0, 1)$：

$$
\boxed{\;S_t = \alpha_t \cdot S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top\;}
$$

- $\alpha_t \to 1$：退化为 DeltaNet（不主动忘）
- $\alpha_t \to 0$：状态清零，开启新话题
- 中间值：让旧记忆按指数衰减

$\alpha_t, \beta_t$ 都是 $x_t$ 的函数，由网络学出来。

这下我们就同时有了：
- **Mamba 风格**的自适应遗忘（通过 $\alpha_t$）
- **DeltaNet 风格**的精确覆盖式 KV 写入（通过 $\beta_t k_t k_t^\top$ 项）

### 4.4 训练并行化：Chunkwise Parallel Form

Gated DeltaNet 的递推比 Linear Attention 复杂（因为 $G_t = \alpha_t(I-\beta_t k_t k_t^\top)$ 不是恒等矩阵），不能直接 cumsum。

实际工程做法：**chunkwise parallel form**——把序列切成块（如每块 64 个 token），块内用并行的稠密计算，块间用串行/扫描。这样既保留了线性总复杂度，又能在 GPU 上跑得飞快。FLA（Flash Linear Attention）库提供了这类高效 kernel。

### 4.5 Qwen3-Next 的混合架构

Gated DeltaNet 在 retrieval 上比 Mamba 强很多，但和 full attention 比仍有差距。所以 Qwen3-Next 的做法是 **混合架构**：

```
[GDN]→[GDN]→[GDN]→[Attn]→[GDN]→[GDN]→[GDN]→[Attn]→ ...
```

3 层 Gated DeltaNet 配 1 层 Full Attention（3:1 比例）。直觉是：

- GDN 层负责长程的高效信息传递（$O(N)$）
- Attention 层负责精确的 in-context 检索（$O(N^2)$ 但只占 1/4 的层数，总开销大幅下降）

这是目前长上下文 LLM 性价比最优的设计之一。

---

## 5. 三者的统一视角

回到第 0 节那个表，现在可以更具体地写出每个模型的 $G_t$：

$$
S_t = G_t S_{t-1} + u_t v_t^\top
$$

| 模型 | $G_t$ | $u_t$ | $v_t$ |
|---|---|---|---|
| Linear Attention | $I$ | $\phi(k_t)$ | $v_t$ |
| Mamba (S6, 单通道) | $\bar A_t$（对角） | $\bar B_t$ | $x_t$（标量化后） |
| Gated DeltaNet | $\alpha_t (I - \beta_t k_t k_t^\top)$ | $\beta_t k_t$ | $v_t$ |

**核心洞察**：所谓"Transformer 之后的下一代序列模型"，本质是在探索"什么样的 $G_t$ 能让一个矩阵状态 RNN 既高效又表达力强"。

近年的设计趋势：
- $G_t = I$（线性注意力，2020）→ 太弱
- $G_t = \mathrm{scalar}\cdot I$（RWKV、RetNet 早期）→ 简单衰减
- $G_t = \mathrm{diag}(\alpha_t)$（Mamba、GLA）→ 通道独立的衰减
- $G_t = \alpha_t(I - \beta_t k_t k_t^\top)$（Gated DeltaNet）→ 有结构的低秩更新
- 还有 TTT、Titans 等：让 $G_t$ 是一个"内层小神经网络"做更复杂的更新

每往 $G_t$ 加一点表达力，就要对应付出一些训练并行化和硬件效率上的代价，这是一直在探索的 trade-off。

---

## 6. 实战建议：什么时候用什么

| 场景 | 推荐 |
|---|---|
| 短序列（< 4k）、对精度要求最高 | Full Attention |
| 中长序列（4k–128k）、需要精确检索 | 混合（Attention + Linear/SSM/GDN） |
| 超长序列（>1M）、流式处理、推理显存敏感 | 纯 Mamba / GDN |
| 想要简单实现、快速实验 | Linear Attention |
| 想要 SOTA 长上下文 LLM | Gated DeltaNet 混合架构（如 Qwen3-Next） |

---

## 7. 进一步阅读路线

如果你想继续深入，可以按这个顺序挖：

1. **数学基础**：HiPPO 论文（理解 SSM 的长程记忆理论）
2. **算法核心**：Mamba 论文中的 Algorithm 1（Selective Scan 的伪代码）
3. **统一视角**：Linear Attention as Fast Weight Programmers（Schlag et al., 2021）—— 把这一族都统一在 fast weight 视角下
4. **在线学习视角**：DeltaNet / TTT / Titans 系列论文
5. **工程实现**：Flash Linear Attention (FLA) 仓库源码

---

## 8. 一页纸总结

```
                        矩阵状态 RNN 家族
                      S_t = G_t S_{t-1} + u_t v_t^T
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
 Linear Attention              Mamba (S6)              Gated DeltaNet
  G_t = I                  G_t = diag(α_t)        G_t = α_t(I - β_t k_t k_t^T)
  纯累加                    选择性 SSM              在线梯度下降 + 遗忘门
  ✗ key collision           ✓ 长序列稳健           ✓ 精确检索 + 自适应遗忘
                            ✗ retrieval 弱          → Qwen3-Next 选用
```

记住这三件事就够了：
1. 它们都是矩阵状态 RNN，区别在状态怎么更新。
2. Linear Attention 是"加"，Mamba 是"按通道选择性记忆"，Gated DeltaNet 是"在线学一个 KV 字典"。
3. 实战中常和 Full Attention 混合使用，取得线性效率 + 强检索能力的平衡。
