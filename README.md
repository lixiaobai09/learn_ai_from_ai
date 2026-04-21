# learn_ai_from_ai

> AI 相关算法与工程原理的学习笔记。所有文档均由作者与AI多轮对话迭代生成，通过AI来学习AI。

## 目录索引

### 1. [Flash Attention 1](./flash_attn_1/)

- [flash_attention_1.md](./flash_attn_1/flash_attention_1.md) — Flash Attention 1 原理详解：从标准 Attention 的显存与带宽瓶颈出发，讲解分块（tiling）、在线 softmax、重计算等核心思想。
- [flash_atten_1.cc](./flash_attn_1/flash_atten_1.cc) — Flash Attention 1 的 C++ 参考实现。

### 2. [注意力机制演进：MHA → MQA → GQA → MLA → DSA](./mha_gqa_mqa_mla_dsa/)

- [attention_variants.md](./mha_gqa_mqa_mla_dsa/attention_variants.md) — 五种注意力（MHA/MQA/GQA/MLA/DSA）变体的「由简入深」讲解，围绕 KV Cache 与推理开销的权衡展开，并做横向对比。