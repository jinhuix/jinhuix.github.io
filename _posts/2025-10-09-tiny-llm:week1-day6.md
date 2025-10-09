---
title: "Tiny-LLM（六）：采样"
date: 2025-10-09 16:46:16 +0800
categories: [默认分类]
tags: []
comments: true
---

## **1 任务总览**

**任务细节**：[Tiny-LLM Week1-Day7](https://skyzh.github.io/tiny-llm/week1-07-sampling-prepare.html)

之前我们已实现了最简单的“贪婪采样”（Greedy Sampling），即每次选择概率最高的 token。本章要扩展三种更灵活的采样方式：

1. **温度采样（Temperature Sampling）**：通过温度参数控制生成的随机性。
2. **Top-k 采样**：只在概率最高的 k 个 token 中采样。
3. **Top-p（核采样, Nucleus Sampling）**：在累积概率超过阈值 p 的 token 集合中采样。

通过这三种方法的组合，模型生成的文本可以在“稳定性”与“创造性”之间灵活平衡，为后续更复杂的推理与解码策略打下基础。

代码文件路径：`src/tiny_llm/sampler.py`



## **2 背景知识**

在语言模型的生成阶段，每一步都会输出一组 token 的概率分布。如何根据这组概率选择下一个 token，是决定模型生成风格的关键。不同的采样策略对应不同的生成特性：

**1. 贪婪采样（Greedy Sampling）**

每次直接选择概率最高的 token，确定性强但缺乏多样性，容易陷入循环或重复输出。例如：

```
输入：Once upon a
输出：time time time ...
```

**2. 温度采样（Temperature Sampling）**

温度参数控制着概率分布的平滑程度：

- **temp=0**：退化为贪婪采样，总是选择概率最高的token
- **temp=1**：保持原始概率分布不变
- **temp>1**：平滑概率分布，让低概率token也有机会被选中，增加创造性
- **temp<1**：锐化概率分布，让高概率token更突出，提高确定性

数学上，它通过将 logits 除以温度实现。温度越高，softmax 分布越平缓，生成结果越具创造性：

```
p_i = softmax(logits_i / temp)
```

**3. Top-k 采样**

在 Top-k 策略中，我们只保留概率最高的 k 个 token，将其余全部屏蔽（概率设为 -∞）。采样只在这 k 个 token 中进行，能有效减少随机性，防止选择到极低概率的 token，同时仍保留一定的多样性。

例如：若 k=10，模型只在前 10 个最有可能的 token 中随机选一个。

**4. Top-p（核采样, Nucleus Sampling）**

Top-p（又称 Nucleus Sampling）是一种自适应的截断策略。不同于固定数量的 Top-k，它会动态选择**累积概率超过阈值 p 的最小 token 集合**。

例如：若 p=0.9，则仅保留累积概率前 90% 的 token，丢弃尾部低概率部分。这种方式在不同上下文下自适应地调整采样空间，效果通常比固定的 Top-k 更自然。

**5. 联合采样（top-k & top-p & Temperature）**

通常我们是将 top-k、top-p、Temperature 联合起来使用。使用的先后顺序是 top-k->top-p->Temperature。

综合来说：

- **Greedy**：确定性高，易重复。
- **Temperature**：控制整体随机性。
- **Top-k / Top-p**：控制采样空间，防止极端输出。



## **3 代码实现**

```python
def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        # 1. top-k
        if top_k is not None and top_k > 0:
            mask_elements = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[:, top_k:]
            logprobs[:, mask_elements] = -mx.inf
        # 2. top-p
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)    # cumulative probs
            mask_elements = cumsum < top_p
            mask_elements[..., 0] = True  # always keep the first one
            logprobs[:, sorted_idx] = mx.where(mask_elements, sorted_logprobs, -mx.inf)
        # 3. temperature scaling
        logprobs = logprobs / temp
        # 4. sample
        return mx.random.categorical(logprobs, axis=-1)

    return sample
```



## **4 参考资料**

- [Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)
- [大模型文本生成——解码策略（Top-k & Top-p & Temperature）](https://www.zhihu.com/tardis/zm/art/647813179?source_id=1003)
