---
title: "cloudlab+greenplum+python3配置"
date: 2025-03-04 20:36:08 +0800
categories: [Database]
tags: [Cloudlab, Greenplum, Python]
comments: true
---

参照论文 ***SWIRL: Selection of Workload-aware Indexes using Reinforcement Learning***  在cloudlab提供的服务器上配置代码所需环境，本地机器为 Windows。

## 1 本地ssh连接远程cloudlab服务器

登录cloudlab官网，注册并租一台服务器。（默认的时长为16小时，注意及时续期，不然服务器上的东西到期会被清理，一般每次续7天）

点击刚刚租好的服务器，打开终端，下面配置本地和远程服务器的ssh连接。

打开本地的.ssh目录（Windows一般是 `C:\User\.ssh` ），里面有个 id_rsa.pub 文件，用记事本的格式打开这个文件，并复制里面的所有内容，即ssh公钥。

接着在远程服务器终端，打开 .ssh目录，里面有个 authorized_keys 文件，将刚刚复制的公钥粘贴到 authorized_keys 里面（从文件内容最后开始粘贴）。

![img1](./assets/img/post/2025-03/cloudlab-1.png)

![img1](./assets/img/post/2025-03/cloudlab-2.png)

再打开vscode，进行远程连接（这里需要装一下远程资源管理器，就是remote development）。点击远程资源管理器的设置按钮，在配置文件中添加host、hostname、user。host可以自己随便取，hostname是远程服务器的ip地址，user是远程服务器的用户名。

![img1](./assets/img/post/2025-03/cloudlab-3.png)

接着就可以在列表看到新连接的服务器了，右键点击 “connect to host in current window”，没问题就连接成功了，可以直接使用。

## 2 在cloudlab上安装greenplum

和在本机上装一样，过程参照：[ubuntu安装greenplum教程及踩坑记录](https://imcaicai.github.io/ubuntu安装greenplum教程及踩坑记录/)

## 3 在cloudlab上安装python3环境

wget工具下载压缩包

```shell
$ wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
$ tar -xzvf Python-3.7.9.tgz
$ ls
gpAdminLogs  gpinitsystem_singlenode  greenplum  hostlist_singlenode  Python-3.7.9  Python-3.7.9.tgz
```

进入解压后的python文件夹，安装python依赖库，并编译安装python

```shell
$ cd Python-3.7.9
# 安装依赖库（非常重要，不然有可能报各种奇奇怪怪的错！）
$ sudo apt-get install libbz2-dev libffi-dev libncurses5-dev libgdbm-dev liblzma-dev sqlite3 libsqlite3-dev openssl libssl-dev tcl8.6-dev tk8.6-dev libreadline-dev zlib1g-dev uuid-dev

# 编译，注意可以在--prefix后指定目录，也可以不指定
$ ./configure --prefix=/usr/local/src/python37
# 安装
$ sudo make && sudo make install
# 链接，注意这里面的/usr/local/src/python37需要换成编译阶段--prefix后面指定的目录
$ sudo ln -s /usr/local/src/python37/bin/python3.7 /usr/bin/python3.7
$ sudo ln -s /usr/local/src/python37/bin/pip3.7 /usr/bin/pip3.7
```

然后需要重新source以下，并重设PYTHONHOME和PYTHONPATH，因为被greenplum改过

```shell
$ source /opt/greenplum-db-6.24.3/greenplum_path.sh
$ unset PYTHONHOME
$ unset PYTHONPATH
```

## 4 运行代码

```shell 
# 创建环境
$ python3.7 -m venv venv
# 激活环境
$ source venv/bin/activate
$ source /opt/greenplum-db-6.24.5/greenplum_path.sh
$ unset PYTHONHOME
$ unset PYTHONPATH

# 安装需要模块（只在第一次安装，后续都不要安装）
# psycopg2-binary==2.8.6不能装
$ pip3 install -r requirements.txt

# 运行
$ python3 -m swirl experiments/tpch.json

# 每次运行前要删除文件：
$ rm -rf experiment_results/ID_TPCH_Test_Experiment
$ psql demo	
\l	# 查看所有数据库
drop database indexselection_tpch___0_01	# 删除数据库
;
\q	# 退出
```

