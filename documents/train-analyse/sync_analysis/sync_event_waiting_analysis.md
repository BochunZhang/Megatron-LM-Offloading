# Sync Event 等待关系分析报告

## 概述

本次分析共识别出 96 个 cudaEventSynchronize 事件。

关键发现：
- **Stream 43 (Forward Offload)** 上记录了 2816 个 Event
- **Stream 47 (Backward Reload)** 上记录了 832 个 Event
- 有 **0** 个 Sync 操作在等待 Stream 43 的 Event
- 有 **0** 个 Sync 操作在等待 Stream 47 的 Event

## Stream 43 (Forward Offload) 等待分析

Stream 43 主要用于前向传播中的激活值卸载 (activation offloading)。

共有 **0** 个同步操作在等待 Stream 43 上的 Event。


## Stream 47 (Backward Reload) 等待分析

Stream 47 主要用于反向传播中的激活值重载 (activation reloading)。

共有 **0** 个同步操作在等待 Stream 47 上的 Event。


## 结论与发现

### 等待关系总结

### 优化建议

1. 等待 Stream 43/47 的 Sync 操作是细粒度 offloading 机制的关键路径
2. 可以通过减少同步频率或增加异步 overlap 来优化性能
3. 关注等待时间较长的 Sync 操作，可能存在优化空间
