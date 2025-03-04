---
title: "ubuntu安装greenplum教程及踩坑记录"
date: 2025-03-04 20:12:29 +0800
categories: [Database]
tags: [Greenplum]
comments: true
---

## 1 版本说明

在撰写本文时，Greenplum 的 Ubuntu 版本是为 Ubuntu 的 18.04 和 16.04 LTS（长期支持）发行版构建的。注意最好用 18.04 版本的ubuntu。（一开始用20及以上的，会有一些奇奇怪怪的错误）

## 2 安装Greenplum

将 Greenplum PPA 存储库添加到 Ubuntu 系统：

```shell
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:greenplum/db
```

更新 Ubuntu 系统以从最近添加的存储库中检索信息：

```shell
sudo apt update
sudo apt install greenplum-db-6
```

上述命令将自动在系统上安装 Greenplum 数据库软件和任何所需的依赖项，并将生成的软件放在 /opt 目录中，如下所示：

![img1](./assets/img/post/2025-03/1.png)

将 Greenplum 数据库软件加载到环境中。注意，应该根据安装的 Greenplum 数据库版本选择 Greenplum 软件目录的确切路径。例如，这里是 `greenplum-db-6.24.3` ：

```shell
source /opt/greenplum-db-6.24.3/greenplum_path.sh
which gpssh
/opt/greenplum-db-6.24.3/bin/gpssh
```

可以通过上面的 `which` 命令测试，看到软件在路径上。现在可以将 Greenplum 集群配置文件模板复制到本地目录中进行编辑，如下所示：

```c++
cp $GPHOME/docs/cli_help/gpconfigs/gpinitsystem_singlenode .
```

## 3 编辑 gpinitsystem 配置文件

创建此文件并仅将主机名放入文件中：

```shell
touch hostlist_singlenode
```

例如我这里是 `xjh-vm` ：

![img2](./assets/img/post/2025-03/2.png)

找一个路径（可以自己随便定）创建名为 master 和 primary 的文件夹，并设置他们的权限。（注意这里是创建文件夹而不是文件！设置权限很重要，否则后面会有报错！）

```shell
mkdir /home/xjh/Desktop/envs/greenplum/master
sudo chmod 777 -R /home/xjh/Desktop/envs/greenplum/master
mkdir /home/xjh/Desktop/envs/greenplum/primary
sudo chmod 777 -R /home/xjh/Desktop/envs/greenplum/primary
```

用 vim 编辑 gpinitsystem_singlenode 文件，主要修改以下4个部分：

```shell
# 这里统一改成这个
MACHINE_LIST_FILE=./hostlist_singlenode
# 这里的路径就是上面创建的文件夹路径，要改成自己设置的路径。注意需要重复2次，中间用空格隔开
declare -a DATA_DIRECTORY=(/home/xjh/Desktop/envs/greenplum/primary /home/xjh/Desktop/envs/greenplum/primary)
# 这里要改成主机名，比如我的是 xjh-vm
MASTER_HOSTNAME=ubuntu
# 这里也要换成自己的路径
MASTER_DIRECTORY=/home/xjh/Desktop/envs/greenplum/master
```

改完就是这个样子的：

![img3](./assets/img/post/2025-03/3.png)

![img4](./assets/img/post/2025-03/4.png)

## 4 运行 gpinit 系统

首先，确保通过运行以下命令交换 ssh 密钥：

```shell
gpssh-exkeys -f hostlist_singlenode
```

接着启动集群：

```shell
gpinitsystem -c gpinitsystem_singlenode
```

下面狂按确认就可以，最后这样就是成功了（过程曲折，太不容易了😭🤧）

![img5](./assets/img/post/2025-03/5.png)

至此可以成功创建数据库：

![img6](./assets/img/post/2025-03/6.png)

## 5 一些踩坑记录

### 5.1 ubuntu版本问题

原来用 20.06 版本的 ubuntu ，出现了一些奇怪的报错，具体忘了。改成 18.04 版本就可以正常安装。

### 5.2 文件权限问题

![img7](./assets/img/post/2025-03/7.png)

需要修改 opt 目录的权限，权限修改方法参照前面的 `sudo chmod 777 -R` 。

### 5.3 文件夹创建问题

![img8](./assets/img/post/2025-03/8.png)

之前的 master 和 primary 都应该是创建的文件夹，第一次创建成文件，就会有这样的报错，找了好久原因。（哭死



参考资料：

[Ubuntu更改镜像源两种方式](https://cloud.tencent.com/developer/article/2030059?from=article.detail.1748201&areaSource=106000.5&traceId=NWIhsf2Zg3rwe3CFYDHef)



