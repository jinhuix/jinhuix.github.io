---
title: "Attention Is All You Need 论文阅读"
date: 2025-09-26 15:03:07 +0800
categories: [LLMs]
tags: [Paper]
comments: true
---

本文为论文 ***Attention Is All You Need*** 的阅读笔记。

## 1 Introduction

**论文核心贡献**

- 提出了一种全新的序列转导模型——**Transformer**，完全基于注意力机制（attention），不再使用循环神经网络（RNN）或卷积网络（CNN）。
- 相比传统的编码器-解码器架构（RNN/LSTM/GRU + attention），Transformer：训练更快（高度可并行化）、翻译质量更高、显著降低训练时间和资源消耗

**背景与动机**

1. 卷积网络序列模型的限制
   - 这些模型中，任意两个位置的依赖计算复杂度随距离增长（ConvS2S 线性，ByteNet 对数），长距离依赖难以学习。
   - Transformer 通过 self-attention 将任意位置依赖的操作数降为常数，并通过 **Multi-Head Attention** 弥补平均加权造成的分辨率降低问题。
2. 注意力机制的优势
   - 能够直接建模序列中任意位置的依赖关系，无需考虑距离。
   - 之前大多数模型都是在 RNN 的基础上加 attention。
3. Transformer 的创新点
   - 完全抛弃循环与卷积，只依赖注意力机制（self-attention）建模序列全局依赖。
   - 支持高度并行化训练，12 小时即可在 8 P100 GPU 上达到优秀效果。



## 2 Model Architecture


