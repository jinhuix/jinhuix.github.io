---
title: "Tiny-LLM（七）：KV Cache"
date: 2025-10-10 16:08:45 +0800
categories: [默认分类]
tags: []
comments: true
---

## **1 任务总览**

**任务细节**：[Tiny-LLM Week2-Day1](https://skyzh.github.io/tiny-llm/week2-01-kv-cache.html)

本节的核心目标是让模型支持 **KV Cache**，实现推理阶段的增量生成，加速“解码”过程。关键任务如下：

1. **实现 TinyKvFullCache 类**（`src/tiny_llm/kv_cache.py`）
   - 提供唯一接口 `update_and_fetch(key, value, mask_length=None, mask=None)`
   - 行为：当缓存为空时初始化；否则沿时间维（序列长度维）拼接新来的 key/value；返回当前完整的 key/value 和当前 offset。
2. **把 Qwen2 多头注意力改为支持缓存**（`src/tiny_llm/qwen2_week2.py`）
   - 每层维护一份自己的缓存实例（每层独立）
   - 模型前向需要接受 offset 参数（表示已处理的 token 数）并断言它与缓存一致性
   - 计算流程：对新输入计算 Q/K/V（Q 长度为 L'，K/V 新增长度为 L'），对 K/V 应用位移（rope/rotary）时使用 offset 范围 slice(offset, offset+L')，然后通过 `cache.update_and_fetch` 拼接并得到完整 K/V（长度为 L），最终用 Q 与缓存 K/V 做注意力计算，仅计算 Q 的最后 L' 行或按需要计算整批 Q 的注意力。
3. **修改模型构造与生成逻辑以使用缓存**（`src/tiny_llm/generate.py`）
   - 模型构造时为每层创建缓存列表（长度 = num_hidden_layers）
   - decode 流程：先 prefill（一次性把 prompt 的 tokens 传入 offset=0），再逐步 decode，每步只把新 token（长度 L'=1）送入并传入 offset=当前已处理长度（旧缓存长度），生成新 token，更新 offset。



## **2 背景知识**

### **2.1 为什么需要 KV Cache？**

在自回归解码中，随着序列增长，每一步都需要计算 `attention(Q, K, V)`。但 QK^T 的上三角（因果掩码）以及早先时间步对应的 **Q 与之前的 K 的乘积在时间上不会改变**——也就是说，对于已经处理过的 token，它们贡献的中间结果可以被重用。KV cache 的目标就是把每层的 K 和 V 保存在内存中，下一次只为新 token 计算新的 K/V 并拼接，attention 只需要对新产生的 Q 与缓存的 K/V 做乘加（或在实现上直接用完整 K/V 但只新增计算量为新增列），从而**把复杂度从 O(L^2) 降到 O(L * L')**（L' 为新 token 数，通常 1）。

### **2.2 重要形状**

```python
L' = new tokens length
L  = total tokens length

update_and_fetch(key, value) -> key, value

key:   B, L', H, D
value: B, L', H, D

self.key   = concat_or_initialize(self.key, key, on the L' dimension)
self.value = concat_or_initialize(self.value, value, on the L' dimension)

self.key:   B, L, H, D
self.value: B, L, H, D

return self.key, self.value
```

其他细节：

- **offset**：当前缓存中已存的序列长度，用来确定 RoPE 位置和后续拼接位置
- **每一层都有自己的 cache**：多层 Transformer 不共享缓存
- **解码过程是循环调用的**：一次只输入一个 token



## **3 代码实现**

### **3.1 实现缓存类 TinyKvFullCache**

关键逻辑：

- 第一次调用：直接存当前 key/value，offset=当前长度
- 后续调用：拼接到 axis=2（序列维度）上
- offset 累加：用于下一次 RoPE 位置和偏移校验
- 返回值格式固定：k, v, offset, mask

```python
class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        N, H, S, D = key.shape  # batch size, head, seq len, dim
        if self.key_values is None:
            self.key_values = (key, value)
            self.offset = S
        else:
            pre_keys, pre_vakues = self.key_values
            new_keys = mx.concat([pre_keys, key], axis=2)
            new_values = mx.concat([pre_vakues, value], axis=2) # shape: (N, H, S+S', D)
            self.key_values = (new_keys, new_values)
            self.offset += S
        return self.key_values[0], self.key_values[1], self.offset, mask
```

### **3.2 实现Qwen2模型**

```python
class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # mlp(x) = ((silu(x * w_gate.T) @ (x * w_up.T))) @ w_down.T
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
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
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. input layer norm
        x_norm = self.input_layernorm(x)

        # 2. self attention
        x = self.self_attn(x_norm, offset, cache, mask) + x

        # 3. post attention layer norm
        x_norm = self.post_attention_layernorm(x)

        # 4. mlp
        x = self.mlp(x_norm) + x

        return x


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
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
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        # 1. embedding
        x = self.embeddings(inputs)  # (N, L, d_model)

        # 2. transformer blocks
        for i, layer in enumerate(self.layers_inner):
            x = layer(x, offset, cache[i], mask="causal")  # (N, L, d_model)
        
        # 3. RMSNorm final
        x = self.norm(x)  # (N, L, d_model)

        # 4. Embedding::as_linear OR Linear (lm_head)
        if self.lm_head is not None:
            return linear(x, self.lm_head)  # untied weights
        else:
            return self.embeddings.as_linear(x) # tied weights
```

### **3.3 实现解码**

关键点：

- 每一层一个 cache
- offset 每次递增 tokens.size
- 解码阶段只输入上一个 token
- logits 只取最后一个 token 的结果
- mask 和 k/v 在模型内部自己拼接处理

```python
def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache) # [batch, seq_len, vocab_size]
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
    offset = 0
    
    # 2. generate/decode
    while True:
        token = _step(model, tokens, offset, kv_cache)  # [batch]
        mx.eval(token)  # ensure computation
        if token.item() == tokenizer.eos_token_id:  # stop if EOS
            break
        detokenizer.add_token(token.item())  # add to detokenizer
        offset += tokens.size
        tokens = token  # only keep the last token for next step
        print(detokenizer.last_segment, end="", flush=True)  # print last segment
```
