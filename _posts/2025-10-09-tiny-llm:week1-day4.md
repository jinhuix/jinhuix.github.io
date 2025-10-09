---
title: "Tiny-LLM（四）：实现RMSNorm与SwiGLU激活的MLP层"
date: 2025-10-09 13:29:14 +0800
categories: [默认分类]
tags: []
comments: true
---

本章将实现 Qwen2 Transformer 架构的两个关键组件：**RMSNorm** 和 **MLP（多层感知器）** 模块，也称为前馈网络。RMSNorm 是一种层归一化技术，与传统的层归一化相比，它有助于以更少的计算开销稳定训练。MLP 模块是一个前馈网络，它处理注意力层的输出，并应用非线性变换来增强模型的表达能力。

## **1 任务总览**

**任务细节**：[Tiny-LLM Week1-Day4](https://skyzh.github.io/tiny-llm/week1-04-rmsnorm-and-mlp.html)

### **1.1 实现RMSNorm**

实现路径：`src/tiny_llm/layer_norm.py`

要求按论文定义实现：对最后一维（每个向量）计算 RMS（均方根），用它来归一化输入并乘以可学习缩放参数 `weight`。实现要注意数值精度（在做平方和、平均与开方时尽量用 `float32`），最后再还原为输入 dtype（如 `float16`）。RMSNorm 定义为：
$$
\text{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2+\epsilon}
\quad,\quad
y=\frac{x}{\text{RMS}(x)}\odot \text{weight}
$$

### **1.2 实现SwiGLU激活的MLP层**  

首先要实现 `basics.py` 中的函数 `silu` 。该函数定义如下：
$$
SiLU(x)=x⋅σ(x)=x/(1+e^{-x})
$$
然后在 `src/tiny_llm/qwen2_week1.py` 中实现 SwiGLU 形式的 MLP（`Qwen2MLP` 类）：
$$
\text{MLP}(x) = \Big(\operatorname{SiLU}(xW_{\text{gate}})\odot (x W_{\text{up}})\Big) W_{\text{down}}
$$


## **2 背景知识**

### **2.1 RMSNorm**

**1. 为什么要用 LayerNorm？**

在神经网络训练中，每一层的输入数据（或者叫激活值）会因为上一层参数的变化而不断波动，导致训练变得不稳定、收敛变慢。 LayerNorm 的作用就是：**把每一层的输入“重新标准化”——让它们的均值变成 0，方差变成 1。**

虽然 LayerNorm 效果好，但它计算很慢。原因是它需要：计算每一层输入的 **均值 (mean)**；再计算 **方差 (variance)**；然后用这两个值对输入进行归一化；最后还要乘上可学习的缩放系数（weight）再加上偏置（bias）。

在深层模型或者循环网络（RNN）中，每层都要这样计算，非常耗时，特别在推理（inference）阶段影响更明显。

<img src="./assets/img/post/2025-10/tinyllm-4-1.png" alt="tinyllm-1"/>

**2. RMSNorm 的动机：去掉“均值”那部分！**

作者发现： LayerNorm的效果主要来自**缩放不变性**，而**重新居中**（减去均值）并非必需。

于是他们大胆提出： **能不能只控制方差（幅度），不计算均值？** 就有了 **RMSNorm（Root Mean Square Normalization，均方根归一化）**。

<img src="./assets/img/post/2025-10/tinyllm-4-2.png" alt="tinyllm-1"/>

虽然它去掉了“减均值”这一步，但实验发现：模型训练依然稳定；收敛速度几乎一样；性能（比如在翻译、图像分类任务中）基本相同；但运行速度能提高 7% 到 64%。

作者还提出了一个更“偷懒”的版本——**pRMSNorm**。它不是用所有输入来计算 RMS，而是只用其中一部分（比如前 6.25%）。如果输入维度是 1000，那它只取前 62 个值来估 RMS。 结果发现这样做几乎不会影响效果，但计算更快了！

### **2.2 MLP与SwiGLU激活**

**MLP**（Multi-Layer Perceptron，多层感知机）本质上是由一系列线性层（线性变换）和非线性激活函数组成的神经网络模块。在 Transformer 或 Qwen2 的架构里：

- 输入：每个位置的 embedding（向量）
- 输出：经过非线性变换后的 embedding
- 作用：在每个 token 上独立提取和组合特征，增加模型表达能力

在 Qwen2 的 MLP 模块中，它使用了 **SwiGLU** 代替普通 ReLU FFN，使得每个位置的向量经过 门控+非线性变换，可以捕捉更复杂的关系。

原始Transformer中的MLP结构：
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
- **SiLU(x)** = x * sigmoid(x)。是一个平滑的激活函数（比 ReLU 更平滑，负区间保留梯度）。
- **GLU**（Gated Linear Unit）类似于门控机制：`(xW1) * sigmoid(xW2)`。
- **SwiGLU**：把 GLU 的 gate 部分换成 SiLU：`SiLU(x W_gate) * (x W_up)`，常见的 MLP 模式是：gate 与 up 两条投影并行计算，逐元素相乘后再通过 `W_down` 投影回 `E`。

SwiGLU 的优势：比 ReLU FFN 有更强表达能力和更好的训练稳定性，是现代 Transformer（包括 Qwen2）常用的 FFN 变体。



## **3 代码实现**

### **3.1 实现RMSNorm**

```python
class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight.astype(mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # y = x / sqrt(mean(x^2) + eps) * weight
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        return (
            self.weight
            * x
            * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        ).astype(orig_dtype)
```

### **3.2 实现SwiGLU激活的MLP层**  

```python
class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # mlp(x) = ((silu(x * w_gate.T) @ (x * w_up.T))) @ w_down.T
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)
```

辅助函数实现：
```python
def silu(x: mx.array) -> mx.array:
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1 + mx.exp(-x))
```



## **4 参考资料**

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)
