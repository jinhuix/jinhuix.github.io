---
title: "PostgreSQL基本用法"
date: 2025-03-04 20:21:58 +0800
categories: [Database]
tags: [PostgreSQL, Greenplum]
comments: true
---

## greenplum服务器

1 进入服务器

```shell
psql username
```

2 创建数据库

```shell
# 方法1：在终端使用createdb命令
createdb databasename

# 方法2：在PostgreSQL内使用命令
create database databasename
```

3 列出所有的数据库

```shell
\l
```

4 进入其中一个数据库

```shell
\c databasename
```

5 返回PostgreSQL主界面

```shell
\q
```

