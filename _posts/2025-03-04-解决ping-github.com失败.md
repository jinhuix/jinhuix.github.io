---
title: "解决ping github.com失败"
date: 2025-03-04 20:04:47 +0800
categories: [Other]
tags: [Git]
pin: false
comments: true
---

## 问题描述

在将本地的远程仓库push到github上时，出现报错：

> ssh: Could not resolve hostname github.com: Temporary failure in name resolution
> fatal: Could not read from remote repository.
>
> Please make sure you have the correct access rights
> and the repository exists.

排除ssh的问题后，在cmd中对目标地址进行ping操作：

```shell
ping github.com
```

出现如下错误提示：

> Ping request could not find host github.com. Please check the name and try again.



## 解决方案

**1 首先获取 github.com IP 地址**

IP 地址查询： [Click](https://ipaddress.com/website/github.com)

通过上述网站查询得到 github.com IP 地址如下

```
140.82.113.4
```

**2 修改hosts文件**

以管理员身份打开本地  `C:\Windows\System32\drivers\etc` 目录下的 hosts 文件，在文件最下方添加：

```
140.82.113.4    github.com
```

完成后保存即可。

**3 再次ping github.com**

此时再次ping github.com即可看到能够成功ping通。

```shell
$ ping 140.82.113.4

Pinging 140.82.113.4 with 32 bytes of data:
Reply from 140.82.113.4: bytes=32 time=229ms TTL=42
Reply from 140.82.113.4: bytes=32 time=229ms TTL=42
Reply from 140.82.113.4: bytes=32 time=229ms TTL=42
Reply from 140.82.113.4: bytes=32 time=229ms TTL=42

Ping statistics for 140.82.113.4:
    Packets: Sent = 4, Received = 4, Lost = 0 (0% loss),
Approximate round trip times in milli-seconds:
    Minimum = 229ms, Maximum = 229ms, Average = 229ms
```

再次进行git push操作，可以顺利执行。