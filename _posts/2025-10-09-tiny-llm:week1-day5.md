---
title: "Tiny-LLM（五）：实现Qwen2模型&生成文本"
date: 2025-10-09 14:01:26 +0800
categories: [默认分类]
tags: []
comments: true
---

在此之前，我们已经实现了多头注意力（MHA）、RMSNorm、RoPE、MLP 等模块。本章将把这些组件整合起来，构建完整的 Qwen2 模型，并让模型“动起来”——让它根据输入的提示（prompt）生成文本。

## **1 任务总览**

**任务细节**：[Tiny-LLM Week1-Day5](https://skyzh.github.io/tiny-llm/week1-05-qwen2-model.html)

**任务细节**：[Tiny-LLM Week1-Day6](https://skyzh.github.io/tiny-llm/week1-06-generate-response.html)

### **1.1 实现 Qwen2TransformerBlock**

文件路径：`src/tiny_llm/qwen2_week1.py`

Qwen2 的 Transformer 块采用如下结构：

```
   input
  /     \
input_layernorm (RMSNorm)
      ↓
Qwen2MultiHeadAttention
      ↓
  Add (残差连接)
  /     \
post_attention_layernorm (RMSNorm)
      ↓
       MLP
      ↓
  Add (残差连接)
      ↓
    output
```

### **1.2 实现 Embedding**

文件路径：`src/tiny_llm/embedding.py`

Embedding 层的作用：

- **编码阶段**：把 token id 映射到向量空间。
- **解码阶段**：可以作为线性层，把隐藏向量映射回词表 logits。

也就是说，在 Qwen2 模型中 Embedding 既是“查表”操作，也是可逆的“线性投影”：

### **1.3 实现 Qwen2ModelWeek1**

文件路径：`src/tiny_llm/qwen2_week1.py`

Qwen2 模型整体结构：

```
input tokens 
    ↓			(tokens: N..)
Embedding 
    ↓			(N.. x hidden_size); note that hidden_size==embedding_dim
多个 Qwen2TransformerBlock
    ↓			(N.. x hidden_size)
RMSNorm
    ↓			(N.. x hidden_size)
Embedding::as_linear 或独立 Linear(lm_head)
    ↓			(N.. x vocab_size)
logits（下一个 token 的分布）
```

模型的输入是一个 token 序列，输出是该序列最后一个 token 的预测概率分布。

### **1.4 实现 `simple_generate`** 

任务主要包括两个部分：

1. **Prefill（预填充）**：将输入的 prompt 作为模型的初始上下文，计算出模型的初始隐状态。
2. **Decode（解码）**：模型在每一步预测下一个 token，将其加入序列中，再送入下一步，直到遇到结束标记（EOS）。



## **2 背景知识**

### **2.1 LLM 是怎么生成文本的？**

大型语言模型（如 GPT、Qwen2）其实并不会直接生成句子。它的每一步仅仅是**预测下一个 token 的概率分布**：

1. Logits计算：模型为词汇表中每个可能的标记分配分数
2. 概率转换：通过softmax将logits转换为概率分布
3. 标记选择：根据概率分布选择下一个标记

例如，输入文本是 "I have a dream"，模型不会直接输出完整下一句话，而是计算一个概率分布， 并且预测 `P(next_token | "I have a dream")`。然后根据解码策略（如贪婪、采样、top-k、temperature）从中选择下一个 token，例如“of”，再拼接到序列末尾。接下来再预测下一个 token，直到生成 `<eos>`（结束标记）。

<img src="./assets/img/post/2025-10/tinyllm-5-1.png" alt="tinyllm-1"/>

文本生成是一个典型的自回归过程：

- 每个新标记的预测都依赖于之前的所有标记
- 序列概率可以分解为条件概率的乘积

```
P(下一个标记 | 已有序列) = softmax(logits)
P(序列) = P(标记₁) × P(标记₂|标记₁) × P(标记₃|标记₁,标记₂) × ...
```

### **2.2 Log-Sum-Exp技巧**

在计算 softmax 时，为避免数值溢出，我们经常使用：

```
logprobs = logits - logsumexp(logits)
```

这一步不会改变相对概率，只是让计算更稳定。



## **3 代码实现**

### **3.1 实现Qwen2TransformerBlock**

`Qwen2TransformerBlock` 是 Transformer 的一个标准编码器/解码器块实现（Qwen2 是在 Transformer 基础上改进的模型），包括：

1. **LayerNorm（或这里的 RMSNorm）**
2. **Self-Attention 子层**
3. **残差连接**
4. **Feed-Forward（MLP）子层**
5. **残差连接**

总体流程为：输入 x → 归一化 → 自注意力 → 残差 → 归一化 → MLP → 残差 → 输出

```
x ──> input_layernorm ──> self_attn ──> + x (residual) ──> h ──> post_attention_layernorm ──> MLP ──> + h (residual) ──> out
```

```python
class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq, wk, wv, wo,
            bq, bk, bv,
            max_seq_len,
            theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. input layer norm
        x_norm = self.input_layernorm(x)

        # 2. self attention
        x = self.self_attn(x_norm, mask) + x

        # 3. post attention layer norm
        x_norm = self.post_attention_layernorm(x)

        # 4. mlp
        x = self.mlp(x_norm) + x

        return x
```

### **3.2 实现Embedding**

```python
class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight    # [vocab_size, embedding_dim]

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x, :]

    def as_linear(self, x: mx.array) -> mx.array:
        # y = x @ weight.T
        return linear(x, self.weight)
```

`as_linear` 用于输出阶段的 logits 计算，等价于词表维度的线性投影。

### **3.3 实现Qwen2ModelWeek1**

整个流程：

1. 输入 tokens → Embedding；
2. 经过多个 Transformer 层；
3. RMSNorm 归一化；
4. 投影回词表维度输出 logits。

输出的 logits 表示每个位置上对所有词的预测概率分布。

```python
class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        precision = mx.float16
        self.hidden_size = mlx_model.args.hidden_size

        # 1. embedding
        self.embeddings = Embedding(
            vocab_size=mlx_model.args.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )

        # 2. transformer blocks
        self.layers_inner = []
        for i in range(mlx_model.args.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj).astype(precision)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj).astype(precision)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj).astype(precision)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj).astype(precision)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj).astype(precision)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj).astype(precision)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj).astype(precision)
            
            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=self.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        
        # 3. RMSNorm final
        self.norm = RMSNorm(
            self.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )

        # 4. Embedding::as_linear OR Linear (lm_head)
        if not mlx_model.args.tie_word_embeddings:
            self.lm_head = dequantize_linear(mlx_model.lm_head).astype(precision)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        # 1. embedding
        x = self.embeddings(inputs)  # (N, L, d_model)

        # 2. transformer blocks
        for layer in self.layers_inner:
            x = layer(x, mask="causal")  # (N, L, d_model)
        
        # 3. RMSNorm final
        x = self.norm(x)  # (N, L, d_model)

        # 4. Embedding::as_linear OR Linear (lm_head)
        if self.lm_head is not None:
            return linear(x, self.lm_head)  # untied weights
        else:
            return self.embeddings.as_linear(x) # tied weights
```

### **3.4 实现 `simple_generate`** 

**_step函数：单步预测**

模型输入形状是 `[batch, seq_len]`，因此加上 `[None]` 代表 batch=1。输出 `logits` 的形状是 `[1, seq_len, vocab_size]`，每个位置都对应预测下一个 token 的概率分布。只取最后一个位置（即当前序列末尾）的 logits，用于预测下一个 token。通过 log-sum-exp 技巧做归一化，然后取概率最高的 token（贪婪策略）。

```python
def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        logits = model(y[None]) # [batch, seq_len, vocab_size]
        logits = logits[:, -1, :]   # last token logits [batch, vocab_size]
        logprobs = logits - mx.logsumexp(logits, keepdims=True) # x - log(sum(exp(x))) for numerical stability
        if sampler is None:
            y = mx.argmax(logprobs, axis=-1)  # greedy
        else:
            y = sampler(logprobs)
        return y

    # 1. prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))  # [seq_len]
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    
    # 2. generate/decode
    while True:
        token = _step(model, tokens)  # [batch]
        mx.eval(token)  # ensure computation
        tokens = mx.concat([tokens, token])  # append new token
        if token.item() == tokenizer.eos_token_id:  # stop if EOS
            break
        detokenizer.add_token(token.item())  # add to detokenizer
        print(detokenizer.last_segment, end="", flush=True)  # print last segment
```



## **4 参考资料**

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [LLM Embeddings Explained: A Visual Primer](https://huggingface.co/spaces/hesamation/primer-llm-embedding)
- [Qwen2.5-7B-Instruct Model Architecture and Parameters](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct?show_file_info=model.safetensors.index.json)
- [The Log-Sum-Exp Trick for Numerical Stability](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- [Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)
