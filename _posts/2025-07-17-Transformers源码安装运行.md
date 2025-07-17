---
title: "Transformers源码安装运行"
date: 2025-07-17 15:07:49 +0800
categories: [大模型]
tags: [大模型推理, Transformers]
comments: true
---

## **环境配置**

可以成功安装运行的搭配：

| 环境    | 版本        |
| ------- | ----------- |
| CUDA    | 11.8        |
| Python  | 3.10.8      |
| vLLM    | 0.6.4.post1 |
| PyTorch | 2.5.1+cu124 |

## **安装运行**

```shell
# clone
git clone https://github.com/huggingface/transformers.git
cd transformers

# 创建虚拟环境
python3 -m venv hfenv
source hfenv/bin/activate

# 安装开发依赖
pip install -e .
```

可以通过以下命令验证是否安装成功：

```shell
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## **踩坑记录**

**1. 无法访问Hugging Face网站**

报错信息：

> Connection to huggingface.co timed out. (connect timeout=10)
> OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.

这是因为国内无法访问 `https://huggingface.co`，可以使用镜像 `https://hf-mirror.com`：

```shell
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

