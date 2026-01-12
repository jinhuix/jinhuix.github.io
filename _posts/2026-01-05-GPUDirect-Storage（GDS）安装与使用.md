---
title: "GPUDirect Storage（GDS）安装与使用"
date: 2026-01-05 00:11:31 +0800
categories: [默认分类]
tags: []
comments: true
---

## **1 GDS 宿主机安装**

### **1.1 环境要求**

需要支持 GDS 的 NVIDIA GPU 和文件系统，具体可查阅官方文档。

### **1.2 安装GDS**

（不确定此处步骤是否有些可省略，仅供参考）

- 挂载iso

[官网](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/)查找对应版本iso，上传至服务器root目录并挂载。

```bash
# 查看nvidia驱动版本
nvidia-smi

# 查看ubunt版本
lsb_release -a

# 挂载iso
sudo -i
mkdir -p /mnt/ofed
mount -o loop /root/MLNX_OFED_LINUX-5.8-7.0.6.1-ubuntu22.04-x86_64.iso /mnt/ofed
cd /mnt/ofed
```

- 补齐编译依赖：

```bash
apt update
apt install -y linux-headers-$(uname -r) build-essential dkms lsb-release python3
```

- 重装OFED驱动：（比较慢，大概10min）

```bash
./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support --without-ucx-cuda

# 更新名称
update-initramfs -u -k `uname -r`
reboot
```

- 查看cuda版本

```bash
nvcc --version

# 如果不对，配置环境变量
vim /etc/bash.bashrc
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source /etc/bash.bashrc

# 再次查看
echo $PATH
echo $LD_LIBRARY_PATH
```

- 解压+编译+加载ko文件：

```bash
cd /root
tar -zxvf gds-nvidia-fs-2.17.4.tar.gz
cd gds-nvidia-fs-2.17.4/
make
insmod nvidia-fs.ko

# 查看是否加载成功
lsmod | grep nvidia_fs
```

- 其他

```bash
# 分别开启nfs和rdma两协议：
modprobe nfs
modprobe rpcrdma

# 电源设置，要求active
systemctl status nvidia-persistenced

# 规则设置
cp /lib/udev/rules.d/40-vm-hotadd.rules /etc/udev/rules.d 
sed -i '/SUBSYSTEM=="memory", ACTION=="add"/d' /etc/udev/rules.d/40-vm-hotadd.rules

# 守护进程：
nvidia-smi -pm 1
```

### **1.3 检查GDS**

安装完成后，验证GDS是否安装成功：

```bash
/usr/local/cuda-<x>.<y>/gds/tools/gdscheck.py -p
```

示例输出：

```bash
xjhui@gpu-2:/usr/local/cuda-12.8/gds/tools$  ./gdscheck -p
warn: error opening log file: Permission denied, logging will be disabled
 GDS release version: 1.13.0.11
 nvidia_fs version:  2.17 libcufile version: 2.12
 Platform: x86_64
 ============
 ENVIRONMENT:
 ============
 =====================
 DRIVER CONFIGURATION:
 =====================
 NVMe P2PDMA        : Unsupported
 NVMe               : Supported
 NVMeOF             : Unsupported
 SCSI               : Unsupported
 ScaleFlux CSD      : Unsupported
 NVMesh             : Unsupported
 DDN EXAScaler      : Unsupported
 IBM Spectrum Scale : Unsupported
 NFS                : Supported
 BeeGFS             : Unsupported
 WekaFS             : Unsupported
 Userspace RDMA     : Unsupported
 --Mellanox PeerDirect : Enabled
 --rdma library        : Not Loaded (libcufile_rdma.so)
 --rdma devices        : Not configured
 --rdma_device_status  : Up: 0 Down: 0
 =====================
 CUFILE CONFIGURATION:
 =====================
 properties.use_pci_p2pdma : false
 properties.use_compat_mode : true
 properties.force_compat_mode : false
 properties.gds_rdma_write_support : true
 properties.use_poll_mode : false
 properties.poll_mode_max_size_kb : 4
 properties.max_batch_io_size : 128
 properties.max_batch_io_timeout_msecs : 5
 properties.max_direct_io_size_kb : 1024
 properties.max_device_cache_size_kb : 131072
 properties.max_device_pinned_mem_size_kb : 18014398509481980
 properties.posix_pool_slab_size_kb : 4 1024 16384 
 properties.posix_pool_slab_count : 128 64 64 
 properties.rdma_peer_affinity_policy : RoundRobin
 properties.rdma_dynamic_routing : 0
 fs.generic.posix_unaligned_writes : false
 fs.lustre.posix_gds_min_kb: 0
 fs.beegfs.posix_gds_min_kb: 0
 fs.weka.rdma_write_support: false
 fs.gpfs.gds_write_support: false
 fs.gpfs.gds_async_support: true
 profile.nvtx : false
 profile.cufile_stats : 0
 miscellaneous.api_check_aggressive : false
 execution.max_io_threads : 0
 execution.max_io_queue_depth : 128
 execution.parallel_io : false
 execution.min_io_threshold_size_kb : 1024
 execution.max_request_parallelism : 0
 properties.force_odirect_mode : false
 properties.prefer_iouring : false
 =========
 GPU INFO:
 =========
 GPU index 0 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 GPU index 1 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 GPU index 2 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 GPU index 3 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 GPU index 4 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 GPU index 5 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 GPU index 6 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 GPU index 7 NVIDIA H100 80GB HBM3 bar:1 bar size (MiB):131072 supports GDS, IOMMU State: Disabled
 ==============
 PLATFORM INFO:
 ==============
 IOMMU: disabled
 Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)
 Cuda Driver Version Installed:  12080
 Platform: R8868 G13, Arch: x86_64(Linux 5.15.0-105-generic)
 Platform verification succeeded
```

有 Supported 则说明 GDS 配置成功，且有支持的文件系统。注意 IOMMU 需要是 disabled。如果发现gds目录下没有tools文件夹那说明toolkit安装有缺失：`apt install --reinstall gds-tools-12-8` 重装即可。IOMMU如果不是disabled，需要禁用。同时要求nvidia-fs版本符合要求。

后续正确添加挂载点即可：

```bash
# 挂载文件
mount -t nfs -o vers=3,nolock,proto=rdma,nconnect=8,port=20049 10.10.10.119:/test0908 /mnt/gds_rdma
# 查看挂载是否成功 rdma/nvme/nfs...
mount | grep rdma
```



## **2 GDS容器配置**

起容器脚本：

```bash
docker run \
    --gpus all \
    --privileged \
    --ipc=host \
    --network=host \
    --cap-add SYS_ADMIN \
    --device /dev/infiniband \	# 需要挂载的 RDMA 字符设备
    --device /dev/nvidia-fs* \	# 用户端所有nvidia-fs
    -v /opt/gds/lib:/usr/local/gds/lib:ro \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \	# sysfs 要挂载的 infiniband 路径
    -v /var/lib/nfsd-ro:/host/nfsd:ro \
    -v /home:/home \
		-v /usr/lib/x86_64-linux-gnu \	# GDS相关so包
    -w /workspace \
    --entrypoint="/bin/bash" \
    --name $2 \
    -itd $1
```

说明：gdscheck 还要读 `/proc/fs/nfsd/exports` 确认 NFS 服务器正在运行；容器默认没有这个挂载点，把接口只读挂进去。但是`-v /proc/fs/nfsd:/proc/fs/nfsd:ro` 大概率会报错，因为Docker 禁止把 /proc 子目录再挂载到容器里（proc-safety 检查），让容器只读宿主机的 /proc/fs/nfsd，用 bind mount + 只读 并且不挂在 /proc 下：

```bash
# 在宿主机先创建一个普通目录做跳板：
mkdir -p /var/lib/nfsd-ro
mount --bind /proc/fs/nfsd /var/lib/nfsd-ro

# 启动容器时挂这个跳板目录，目标路径不要放在 /proc：（即添加这行指令）
-v /var/lib/nfsd-ro:/host/nfsd:ro
```

通过脚本启动自己的gds容器，并配置剩余内容：

```bash
# 启动容器
sh start_gds_container.sh 镜像ID test-gds

# 把tools拷进容器
docker cp /usr/local/cuda-12.8/gds/tools test-gds:/usr/local/cuda-12.8/gds/

# 加可执行权限
docker exec fenghao-gds chmod +x /usr/local/cuda-12.8/gds/tools/*

# 配置环境变量
vim /etc/bash.bashrc
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source /etc/bash.bashrc

# 检查环境变量：
echo $PATH
echo $LD_LIBRARY_PATH
```

检查GDS：

```bash
# 检查GDS
/usr/local/cuda-<x>.<y>/gds/tools/gdscheck.py -p

# 检查挂载点
cat /proc/mounts | grep rdma
```

注意一定要在宿主机先挂好，如果宿主机挂载点有变化记得`docker restart` 。



## **3 GDSIO测试**

`/usr/local/cuda-12.8/gds/tools/gdsio` 提供了 8 个不同的测试模式，分别是：

| **传输模式** | **传输类型 (XferType)** | **含义**                                       |
| ------------ | ----------------------- | ---------------------------------------------- |
| 0            | GPUD                    | 数据直接从存储传输到 GPU，绕过 CPU             |
| 1            | CPUONLY                 | 数据从存储传输到 CPU                           |
| 2            | CPU_GPU                 | 数据先从存储传输到 CPU，经过处理后再传输到 GPU |
| 3            | CPU_ASYNC_GPU           | 数据先从存储传输到 CPU，然后异步传输到 GPU     |
| 4            | CPU_CACHED_GPU          | 存储->页缓存->CPU->GPU                         |
| 5            | ASYNC                   | 数据直接从存储异步传输到 GPU                   |
| 6            | GPU_BATCH               | 存储到 GPU 的批处理传输                        |
| 7            | GPU_BATCH_STREAM        | 存储到 GPU 的流式批处理传输                    |

测试命令如下：

```bash
/usr/local/cuda-12.8/gds/tools/gdsio -f /data/test/dd.txt -d 0 -w 4 -s 10G -i 1M -x 2 
IoType: READ XferType: CPU_GPU Threads: 4 DataSetSize: 10142720/10240000(KiB) IOSize: 1024(KiB) Throughput: 4.312804 GiB/sec，Avg_Latency: 905.410136 usecs ops: 9905 total_time 2.242822 secs 
```

参数含义为：-d 设备号，-w 线程数，-s 文件大小，-i 单次I/O大小，-x 传输模式



## **4 如何验证真正使用了GDS？**

### **4.1 用 `gdscheck` 证明挂载点/路径是 GDS-capable**

```bash
# 查系统上哪些 FS 支持 GDS：
sudo /usr/local/cuda-*/gds/tools/gdscheck -p

# 检查实际的 gds_path
sudo /usr/local/cuda-*/gds/tools/gdscheck -p <YOUR_GDS_PATH>
```

### **4.2 看libcufile 日志，确认没有 fallback**

因为 libcufile 可能内部 fallback，常用做法是设置 cuFile 日志环境变量：设置 `compat_mode`  为`false`，跑完后看到明确的 “using nvidia-fs / gds path / GDS enabled” 且没有 “fallback to POSIX” 之类的字样：强证据。如果日志里出现 “POSIX fallback”：那就证明数据面不是纯 GDS。

## **参考资料**

- [GPUDirect Storage的部署与启用条件需求](https://www.ithome.com.tw/tech/165668)
- [GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#example-display-tracepoints)
- [GPUDirect Storage Release Notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
- [Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#cufiledriveropen)
- [https://github.com/NVIDIA/MagnumIO/blob/main/gds/docker/gds-run-container](https://github.com/NVIDIA/MagnumIO/blob/main/gds/docker/gds-run-container)
- [https://www.chenshaowen.com/blog/how-to-enable-gds-on-gpu-host.html](https://www.chenshaowen.com/blog/how-to-enable-gds-on-gpu-host.html)
