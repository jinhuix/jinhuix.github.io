#!/bin/bash

# 获取当前日期
today=$(date +"%Y-%m-%d")
filename="_posts/${today}-${1// /-}.md"

# 创建 Markdown 文件
cat > "$filename" <<EOF
---
title: "$1"
date: $(date +"%Y-%m-%d %H:%M:%S %z")
categories: [默认分类]
tags: []
comments: true
---

这里是你的正文内容。
EOF

echo "新博客文件已创建: $filename"
