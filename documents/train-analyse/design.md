# NSYS Profile Analyzer 设计文档

## 1. 设计目标

- 分析 Megatron-LM 训练过程中的 GPU 性能
- 特别关注 FineGrainedOffloading 的内存拷贝操作
- 支持多进程分布式训练分析（DP=4）
- 按 CUDA Stream 构建独立的执行时间树

## 2. 核心数据结构

### 2.1 BaseEvent (基类)
```python
@dataclass
class BaseEvent:
    cpu_start_ns: Optional[int] = None
    cpu_end_ns: Optional[int] = None
    cpu_duration_ns: Optional[int] = None
    gpu_start_ns: Optional[int] = None
    gpu_end_ns: Optional[int] = None
    gpu_duration_ns: Optional[int] = None
    process_signature: int = 0
    thread_type: ThreadType = ThreadType.UNKNOWN
    is_main_thread: bool = False
    parent: Optional['BaseEvent'] = None
    children: List['BaseEvent'] = field(default_factory=list)
```

### 2.2 CudaEvent
```python
@dataclass
class CudaEvent(BaseEvent):
    device_id: Optional[int] = None
    stream_id: Optional[str] = None
    correlation_id: int = 0
    global_pid: Optional[str] = None
    kernel_name: Optional[str] = None
    is_memcpy: bool = False
    memcpy_type: Optional[str] = None
```

### 2.3 NvtxEvent
```python
@dataclass
class NvtxEvent(BaseEvent):
    text: str = ""
    global_tid: Optional[str] = None
    iteration: Optional[int] = None
    step_type: StepType = StepType.UNKNOWN
    # Stream-level tree structure
    stream_gpu_time: Dict[str, Tuple[Optional[int], Optional[int], int]]
    stream_parent: Dict[str, 'NvtxEvent']
    stream_children: Dict[str, List['NvtxEvent']]
```

## 3. TID/PID 结构

64-bit GlobalTid/GlobalPid 结构：
```
0x0001 0a16ce 0a16ce
│      │       │
│      │       └─ 低 24 bits: 线程标识
│      └─ 中 24 bits: 进程标识
└─ 高 16 bits: 固定前缀 (0x0001)
```

**特征值计算**: `(high_16 << 24) | mid_24` (高40bits)

## 4. 分析流程

1. **设备信息加载**: 从 SQLite 数据库读取 GPU 设备信息
2. **事件解析**: 解析 NVTX Event、TraceProcessEvent、CudaEvent
3. **事件关联**: 使用 correlation_id 关联 TraceProcessEvent 和 CudaEvent
4. **CPU 树形构建**: 基于时间交叠构建父子关系
5. **Stream GPU 时间计算**: 递归收集子 CudaEvent，按 stream 分组
6. **Stream 树形构建**: 为每个 stream 构建独立的执行时间树
7. **Step 分析**: 计算 forward_step、backward_step、optimizer_step 的 GPU 时间
8. **Offloading 分析**: 分析 FineGrainedOffloading 操作

## 5. 输出格式

### 5.1 JSON 输出
- 完整事件树: `iteration_{N}_{step_name}_{pcie_bus}_v3.json`
- Stream 树: `iteration_{N}_{step_name}_stream_{stream_id}_{pcie_bus}_v3.json`

### 5.2 Excel 输出
- 文件名: `iteration_{N}_gpu_times_{pcie_bus}.xlsx`
- Summary 工作表: 总 GPU 执行时间
- Per-stream 工作表: 每个 Stream 的详细时间

## 6. 使用方式

### 6.1 Python 脚本
```bash
python tests/train_analyse/nsys_profile_analyzer.py \
    --sqlite profile.sqlite \
    --json profile.json \
    --output ./results \
    --iteration 16
```

### 6.2 Shell 脚本
```bash
./tests/train_analyse/run.sh --iteration 16 logs/nsys-profile/
```

### 6.3 DeepSeek 训练脚本
```bash
# 启用对比模式
./examples/deepseek-v3/train_deepseek_v3_gb200.sh \
    --compare-fine-grained-offload \
    --fine-grained-offload
```

## 7. 文件位置

- 分析器代码: `tests/train_analyse/nsys_profile_analyzer.py`
- 运行脚本: `tests/train_analyse/run.sh`
- 设计文档: `documents/train_analyse/design.md`
- 示例分析结果: `documents/train_analyse/example/`
