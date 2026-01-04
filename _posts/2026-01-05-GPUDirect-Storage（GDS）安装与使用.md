---
title: "GPUDirect Storage（GDS）安装与使用"
date: 2026-01-05 00:11:31 +0800
categories: [默认分类]
tags: []
comments: true
---

## **1 GDS 安装**

### **1.1 在宿主机上配置**

在宿主机上的环境配置及安装可以参考以下文章：

- [GPUDirect Storage的部署与启用条件需求](https://www.ithome.com.tw/tech/165668)
- [GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#example-display-tracepoints)
- [GPUDirect Storage Release Notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
- [Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#cufiledriveropen)

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

有 Supported 则说明 GDS 配置成功吗，且有支持的文件系统。注意 IOMMU 需要是 disabled。

接着挂载文件，例如：

```bash
# 挂载文件
mount -t nfs -o vers=3,nolock,proto=rdma,nconnect=8,port=20049 10.10.10.119:/test0908 /mnt/gds_rdma
# 查看挂载是否成功 rdma/nvme/nfs...
mount | grep rdma
```

### **1.2 在容器上配置**

[https://github.com/NVIDIA/MagnumIO/blob/main/gds/docker/gds-run-container](https://github.com/NVIDIA/MagnumIO/blob/main/gds/docker/gds-run-container)

在宿主机上安装好GDS后，一般会在容器中使用。在执行docker run的时候同样要配置一些参数。

```bash
docker run \
    --gpus all \
    --ipc=host \
    --network=host \
    --ulimit memlock=-1 \
    --privileged \
    --cap-add SYS_ADMIN \
    --cap-add=IPC_LOCK \
    --device /dev/infiniband/rdma_cm \
    --device /dev/infiniband/uverbs* \
    --device /dev/nvidia-fs* \
    --device /dev/infiniband/umad* \
    -v /usr/lib/x86_64-linux-gnu/libmlx5.so.1:/usr/lib/x86_64-linux-gnu/libmlx5.so.1:ro \
    -v /usr/lib/x86_64-linux-gnu/libibverbs.so.1:/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro \
    -v /usr/lib/x86_64-linux-gnu/librdmacm.so.1:/usr/lib/x86_64-linux-gnu/librdmacm.so.1:ro \
    -v /usr/lib/x86_64-linux-gnu/rsocket/librspreload.so.1:/usr/lib/x86_64-linux-gnu/rsocket/librspreload.so.1:ro \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \
    -v /sys/class/net:/sys/class/net:ro \
    -v /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufile.so.0:/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0:ro \
    -v /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufile_rdma.so.1:/usr/local/cuda/targets/x86_64-linux/lib/libcufile_rdma.so.1:ro \
    -v /var/lib/nfsd-ro:/host/nfsd:ro \
    -v /home/models:/home/models \
    -v /home:/home \
    -v /home/nfs:/home/nfs:rw,rshared \
    -w /workspace \
    --entrypoint="/bin/bash" \
    --name $2 \
    -itd $1
```



## **2 GDS 使用**







## **3 如何验证真正使用了GDS？**

### **3.1 用 `gdscheck` 证明挂载点/路径是 GDS-capable**

```bash
# 查系统上哪些 FS 支持 GDS：
sudo /usr/local/cuda-*/gds/tools/gdscheck -p

# 检查实际的 gds_path
sudo /usr/local/cuda-*/gds/tools/gdscheck -p <YOUR_GDS_PATH>
```

### **3.2 看libcufile 日志，确认没有 fallback**

因为 libcufile 可能内部 fallback，常用做法是设置 cuFile 日志环境变量：设置 compat_mode  为false，跑完后看到明确的 “using nvidia-fs / gds path / GDS enabled” 且没有 “fallback to POSIX” 之类的字样：强证据。如果日志里出现 “POSIX fallback”：那就证明数据面不是纯 GDS。

### **3.3 看 `nvidia-fs` 内核统计计数是否增长**

真正走 GDS，会经过 `nvidia-fs`（kernel module）。可以在跑推理前后各读一次 `nvidia-fs` 的统计计数，确认读/写请求计数在增长。不同版本路径可能不同，常见候选：

```bash
ls /proc/driver | grep -i nvidia
ls /proc/driver/nvidia-fs 2>/dev/null
find /proc/driver -maxdepth 2 -type f | grep -i 'nvidia.*fs'
```

如果存在类似：

- `/proc/driver/nvidia-fs/stats`
- `/proc/driver/nvidia-fs/metrics`
- `/proc/driver/nvidia-fs/*`

就：

```bash
cat /proc/driver/nvidia-fs/stats
# 跑一次 workload
cat /proc/driver/nvidia-fs/stats
```

workload 前后，stats 里的 read/write 相关计数明显增加：证明走了 nvidia-fs 通路（GDS 数据面）。

## **参考资料**

- [GPUDirect Storage的部署与启用条件需求](https://www.ithome.com.tw/tech/165668)
- [GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#example-display-tracepoints)
- [GPUDirect Storage Release Notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
- [Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#cufiledriveropen)
- [https://github.com/NVIDIA/MagnumIO/blob/main/gds/docker/gds-run-container](https://github.com/NVIDIA/MagnumIO/blob/main/gds/docker/gds-run-container)
