#!/usr/bin/env python3
"""
NSYS Profile Analyzer V3
根据用户要求的数据结构和分析逻辑进行设计

Usage:
    python nsys_profile_analyzer.py --sqlite <sqlite_path> --json <json_path> --output <output_dir>
    python nsys_profile_analyzer.py -s <sqlite_path> -j <json_path> -o <output_dir>
"""

import argparse
import json
import sys
import re
import sqlite3
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any
from enum import Enum

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# ============== 常量定义 ==============

TYPE_TRACE_PROCESS_47 = 47
TYPE_TRACE_PROCESS_48 = 48
TYPE_NVTX_59 = 59
TYPE_NVTX_75 = 75
TYPE_CUDA_79 = 79
TYPE_CUDA_80 = 80
TYPE_CUDA_106 = 106


class ThreadType(Enum):
    MAIN = "main"
    AUTOGRAD = "autograd"
    UNKNOWN = "unknown"


class StepType(Enum):
    FORWARD_BACKWARD = "forward_backward"
    FORWARD = "forward_step"
    BACKWARD = "backward_step"
    OPTIMIZER = "optimizer_step"
    ITERATION = "iteration"
    OFFLOAD_COMMIT = "offload_commit"
    OFFLOAD_START = "offload_start"
    UNKNOWN = "unknown"


# ============== 数据类定义 ==============

@dataclass
class DeviceInfo:
    """GPU 设备信息"""
    device_id: int           # CUDA device id
    nsys_gpu_id: int        # NSYS internal GPU id
    name: str = ""
    pcie_bus: str = ""      # PCIe bus location, e.g., "0009:01:00.0"
    properties: Dict = field(default_factory=dict)


@dataclass
class BaseEvent:
    """Event 基类 - 只保留专用时间字段"""
    # CPU 执行区间
    cpu_start_ns: Optional[int] = None
    cpu_end_ns: Optional[int] = None
    cpu_duration_ns: Optional[int] = None

    # GPU 执行区间
    gpu_start_ns: Optional[int] = None
    gpu_end_ns: Optional[int] = None
    gpu_duration_ns: Optional[int] = None

    # 特征值 (用于区分进程) - 高40bit
    process_signature: int = 0

    # 进程信息
    thread_type: ThreadType = ThreadType.UNKNOWN
    is_main_thread: bool = False

    # 树形结构
    parent: Optional['BaseEvent'] = None
    children: List['BaseEvent'] = field(default_factory=list)

    def get_sort_key(self) -> int:
        """获取用于排序的时间戳（优先 CPU，其次 GPU）"""
        if self.cpu_start_ns is not None:
            return self.cpu_start_ns
        if self.gpu_start_ns is not None:
            return self.gpu_start_ns
        return 0

    def has_cpu_time(self) -> bool:
        """检查是否有 CPU 时间"""
        return self.cpu_start_ns is not None


@dataclass
class CudaEvent(BaseEvent):
    """CUDA Event - 包含 TraceProcessEvent 和 CudaEvent 的合并信息"""
    device_id: Optional[int] = None
    stream_id: Optional[str] = None
    correlation_id: int = 0
    global_pid: Optional[str] = None
    global_tid: Optional[str] = None
    kernel_name: Optional[str] = None
    short_name: Optional[str] = None
    demangled_name: Optional[str] = None
    is_memcpy: bool = False
    memcpy_type: Optional[str] = None
    memcpy_size_bytes: Optional[int] = None
    throughput_gbps: Optional[float] = None
    is_sync: bool = False  # 是否为同步事件 (cudaStreamWaitEvent 等)
    sync_type: Optional[str] = None  # 同步类型
    raw_data: Dict = field(default_factory=dict)

    def __post_init__(self):
        """计算派生字段"""
        if self.gpu_start_ns is not None and self.gpu_end_ns is not None:
            self.gpu_duration_ns = self.gpu_end_ns - self.gpu_start_ns
        if self.cpu_start_ns is not None and self.cpu_end_ns is not None:
            self.cpu_duration_ns = self.cpu_end_ns - self.cpu_start_ns


@dataclass
class NvtxEvent(BaseEvent):
    """NVTX Event - CPU 上的 Python 事件记录"""
    text: str = ""
    global_tid: Optional[str] = None
    domain_id: str = "0"
    correlation_ids: Set[int] = field(default_factory=set)
    cuda_events: List[CudaEvent] = field(default_factory=list)
    iteration: Optional[int] = None
    step_type: StepType = StepType.UNKNOWN
    nvtx_type: int = TYPE_NVTX_59
    raw_data: Dict = field(default_factory=dict)
    sizes: List[List[int]] = field(default_factory=list)

    # Stream-level tree structure
    # stream_gpu_time: {stream_id: (gpu_start_ns, gpu_end_ns, gpu_duration_ns)}
    stream_gpu_time: Dict[str, Tuple[Optional[int], Optional[int], int]] = field(default_factory=dict)
    # stream_parent: {stream_id: NvtxEvent}
    stream_parent: Dict[str, 'NvtxEvent'] = field(default_factory=dict)
    # stream_children: {stream_id: List[NvtxEvent]}
    stream_children: Dict[str, List['NvtxEvent']] = field(default_factory=dict)


@dataclass
class StepGPUExecution:
    """记录 Step 的 GPU 执行时间"""
    iteration: int
    step_type: StepType
    process_signature: int
    first_cuda_start_ns: Optional[int] = None
    last_cuda_end_ns: Optional[int] = None
    gpu_duration_ns: int = 0
    cuda_events: List[CudaEvent] = field(default_factory=list)
    nvtx_event: Optional[NvtxEvent] = None


@dataclass
class CopyOperation:
    """Memory Copy 操作分析"""
    iteration: int
    step_type: StepType
    process_signature: int
    parent_nvtx: Optional[NvtxEvent] = None
    module_name: str = ""
    operation_type: str = ""
    operation_desc: str = ""
    copy_count: int = 0
    copy_nvtx_events: List[NvtxEvent] = field(default_factory=list)
    sizes_per_copy: List[List[List[int]]] = field(default_factory=list)
    total_bytes: int = 0
    memcpy_events: List[CudaEvent] = field(default_factory=list)
    memcpy_total_bytes: int = 0
    throughput_gbps: float = 0.0
    is_blocking: bool = False
    gpu_start_ns: Optional[int] = None
    gpu_end_ns: Optional[int] = None


# ============== 辅助函数 ==============

def parse_tid_structure(tid: int) -> Dict:
    high_16 = (tid >> 48) & 0xFFFF
    mid_24 = (tid >> 24) & 0xFFFFFF
    low_24 = tid & 0xFFFFFF
    is_main = low_24 == mid_24
    thread_type = ThreadType.MAIN if is_main else ThreadType.AUTOGRAD
    return {
        "high_16": high_16,
        "process_id": mid_24,
        "thread_id": low_24,
        "is_main_thread": is_main,
        "thread_type": thread_type,
        "process_signature": (high_16 << 24) | mid_24,
    }


def get_process_signature_from_id(tid_or_pid: int) -> int:
    return (tid_or_pid >> 24) & 0xFFFFFFFFFF


def extract_step_type(text: str) -> StepType:
    text_lower = text.lower()
    if "iteration" in text_lower and "=" not in text_lower:
        return StepType.ITERATION
    elif "forward_backward_func" in text_lower:
        return StepType.FORWARD_BACKWARD
    elif "forward_step" in text_lower:
        return StepType.FORWARD
    elif "backward_step" in text_lower:
        return StepType.BACKWARD
    elif "megatron.training.training.train_step.optimizer.step" in text_lower:
        # Only match the top-level optimizer step, not internal ones
        return StepType.OPTIMIZER
    elif "FineGrainedOffloadingGroupCommitFunction" in text:
        return StepType.OFFLOAD_COMMIT
    elif "FineGrainedOffloadingGroupStartFunction" in text:
        return StepType.OFFLOAD_START
    return StepType.UNKNOWN


def simplify_step_name(text: str, step_type: StepType) -> str:
    """简化 step 名称

    Examples:
        megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining.forward_step[0] -> forward_step[0]
        megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining.backward_step[0] -> backward_step[0]
        megatron.training.training.train_step.optimizer.step -> optimizer_step
    """
    if step_type == StepType.FORWARD:
        # 提取 forward_step[0]
        match = re.search(r'forward_step\[\d+\]', text)
        if match:
            return match.group(0)
        return "forward_step"
    elif step_type == StepType.BACKWARD:
        # 提取 backward_step[0]
        match = re.search(r'backward_step\[\d+\]', text)
        if match:
            return match.group(0)
        return "backward_step"
    elif step_type == StepType.OPTIMIZER:
        # 简化为 optimizer_step
        return "optimizer_step"
    return text


def extract_iteration(text: str) -> Optional[int]:
    match = re.search(r'iteration\s+(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'it[=:]\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_sizes_from_text(text: str) -> List[List[int]]:
    match = re.search(r'sizes\s*=\s*(\[.+?\])', text)
    if match:
        try:
            sizes_str = match.group(1)
            return json.loads(sizes_str.replace("'", '"'))
        except:
            pass
    return []


def calculate_tensor_bytes(sizes: List[List[int]], dtype_size: int = 4) -> int:
    total = 0
    for size_list in sizes:
        if size_list:
            prod = 1
            for dim in size_list:
                prod *= dim
            total += prod * dtype_size
    return total


# ============== 主分析器类 ==============

class NSYSAnalyzer:
    def __init__(self, json_filepath: str, sqlite_filepath: str, output_dir: str, iteration: int = 16):
        self.json_filepath = json_filepath
        self.sqlite_filepath = sqlite_filepath
        self.output_dir = output_dir
        self.iteration = iteration
        self.devices: Dict[int, DeviceInfo] = {}  # key: cuda device id
        self.nvtx_events: List[NvtxEvent] = []
        self.cuda_events_by_sig_corr: Dict[int, Dict[int, CudaEvent]] = defaultdict(dict)
        self.cuda_events_by_signature: Dict[int, List[CudaEvent]] = defaultdict(list)
        self.cuda_events_by_stream: Dict[str, List[CudaEvent]] = defaultdict(list)
        self.nvtx_by_signature: Dict[int, List[NvtxEvent]] = defaultdict(list)
        self.events_by_signature: Dict[int, List[BaseEvent]] = defaultdict(list)
        self.root_events: Dict[int, List[BaseEvent]] = defaultdict(list)
        self.step_executions: List[StepGPUExecution] = []
        self.copy_operations: List[CopyOperation] = []
        self.bugs: List[str] = []
        self.iterations: Dict[int, List[NvtxEvent]] = defaultdict(list)
        self.string_table: Dict[str, str] = {}
        self.sig_to_device: Dict[int, int] = {}  # process_signature -> main device_id

    def log_bug(self, message: str):
        self.bugs.append(message)

    def resolve_string(self, name_id: str) -> str:
        if name_id in self.string_table:
            return self.string_table[name_id]
        return name_id

    def load_devices_from_sqlite(self):
        """从 sqlite 数据库加载 GPU 设备信息"""
        print("=" * 80)
        print("Loading device info from SQLite")
        print("=" * 80)

        conn = sqlite3.connect(self.sqlite_filepath)
        cursor = conn.cursor()

        # 从 TARGET_INFO_GPU 表读取设备信息
        cursor.execute('SELECT id, name, busLocation, cuDevice FROM TARGET_INFO_GPU')
        rows = cursor.fetchall()

        for row in rows:
            nsys_gpu_id, name, bus_location, cu_device = row
            device_info = DeviceInfo(
                device_id=cu_device,      # CUDA device id
                nsys_gpu_id=nsys_gpu_id,
                name=name,
                pcie_bus=bus_location     # e.g., "0009:01:00.0"
            )
            self.devices[cu_device] = device_info
            print(f"  Device {cu_device}: {name}, PCIe={bus_location}")

        conn.close()

    def load_and_parse(self):
        print("\n" + "=" * 80)
        print("Step 1-4: Loading and parsing events")
        print("=" * 80)

        line_count = 0
        nvtx_count = 0
        cuda_api_count = 0
        cuda_kernel_count = 0

        with open(self.json_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    event_type = record.get("Type")

                    if "id" in record and "value" in record:
                        self.string_table[record["id"]] = record["value"]

                    if event_type == TYPE_NVTX_59 or event_type == TYPE_NVTX_75:
                        self._parse_nvtx_event(record, event_type)
                        nvtx_count += 1
                    elif event_type in (TYPE_TRACE_PROCESS_47, TYPE_TRACE_PROCESS_48):
                        self._parse_trace_process_event(record)
                        cuda_api_count += 1
                    elif event_type in (TYPE_CUDA_79, TYPE_CUDA_80, TYPE_CUDA_106):
                        self._parse_cuda_event(record, event_type)
                        cuda_kernel_count += 1
                except json.JSONDecodeError:
                    pass

                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count} lines...")

        print(f"\nTotal lines: {line_count}")
        print(f"NVTX events: {nvtx_count}")
        print(f"CUDA API events: {cuda_api_count}")
        print(f"CUDA kernel events: {cuda_kernel_count}")
        print(f"Unique process signatures: {len(self.cuda_events_by_sig_corr)}")

    def _build_sig_to_device_mapping(self):
        """Build mapping from process_signature to main device_id based on CUDA events"""
        print("\n" + "=" * 80)
        print("Building process_signature to device mapping")
        print("=" * 80)

        sig_device_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for sig, cuda_events in self.cuda_events_by_signature.items():
            for evt in cuda_events:
                if evt.device_id is not None and evt.device_id >= 0:
                    sig_device_counts[sig][evt.device_id] += 1

        for sig, device_counts in sig_device_counts.items():
            if device_counts:
                # Find device with most events
                main_device = max(device_counts.items(), key=lambda x: x[1])[0]
                self.sig_to_device[sig] = main_device

        print(f"Built mapping for {len(self.sig_to_device)} process signatures:")
        for sig, device_id in sorted(self.sig_to_device.items()):
            devices = sig_device_counts[sig]
            print(f"  Sig {sig} -> Device {device_id} {dict(devices)}")

    def _parse_nvtx_event(self, record: Dict, nvtx_type: int):
        nvtx_data = record.get("NvtxEvent", {})
        global_tid_str = nvtx_data.get("GlobalTid", "0")
        global_tid = int(global_tid_str)
        tid_info = parse_tid_structure(global_tid)

        text = nvtx_data.get("Text", "")
        start_ns = int(nvtx_data.get("Timestamp", "0"))
        end_ns = int(nvtx_data.get("EndTimestamp", "0")) if "EndTimestamp" in nvtx_data else start_ns
        sizes = extract_sizes_from_text(text)
        iteration = extract_iteration(text)

        nvtx_event = NvtxEvent(
            cpu_start_ns=start_ns,
            cpu_end_ns=end_ns,
            cpu_duration_ns=end_ns - start_ns,
            process_signature=tid_info["process_signature"],
            thread_type=tid_info["thread_type"],
            is_main_thread=tid_info["is_main_thread"],
            text=text,
            global_tid=global_tid_str,
            domain_id=str(nvtx_data.get("DomainId", "0")),
            iteration=iteration,
            step_type=extract_step_type(text),
            nvtx_type=nvtx_type,
            raw_data=nvtx_data,
            sizes=sizes
        )

        self.nvtx_events.append(nvtx_event)
        self.nvtx_by_signature[tid_info["process_signature"]].append(nvtx_event)

        if iteration is not None:
            self.iterations[iteration].append(nvtx_event)

    def _parse_trace_process_event(self, record: Dict):
        trace_data = record.get("TraceProcessEvent", {})
        global_tid_str = trace_data.get("globalTid", "0")
        global_tid = int(global_tid_str)
        tid_info = parse_tid_structure(global_tid)

        correlation_id = int(trace_data.get("correlationId", 0))
        start_ns = int(trace_data.get("startNs", "0"))
        end_ns = int(trace_data.get("endNs", "0"))

        sig = tid_info["process_signature"]

        if correlation_id in self.cuda_events_by_sig_corr[sig]:
            existing = self.cuda_events_by_sig_corr[sig][correlation_id]
            if existing.cpu_start_ns is not None and correlation_id != 0:
                self.log_bug(f"Duplicate TraceProcessEvent for correlation_id={correlation_id}, sig={sig}")
            else:
                existing.cpu_start_ns = start_ns
                existing.cpu_end_ns = end_ns
                existing.cpu_duration_ns = end_ns - start_ns
                existing.global_tid = global_tid_str
        else:
            cuda_event = CudaEvent(
                cpu_start_ns=start_ns,
                cpu_end_ns=end_ns,
                cpu_duration_ns=end_ns - start_ns,
                process_signature=sig,
                thread_type=tid_info["thread_type"],
                is_main_thread=tid_info["is_main_thread"],
                correlation_id=correlation_id,
                global_pid=None,
                global_tid=global_tid_str,
                raw_data=trace_data
            )
            self.cuda_events_by_sig_corr[sig][correlation_id] = cuda_event

    def _parse_cuda_event(self, record: Dict, event_type: int):
        cuda_data = record.get("CudaEvent", {})
        global_pid_str = cuda_data.get("globalPid", "0")
        global_pid = int(global_pid_str)
        sig = get_process_signature_from_id(global_pid)

        correlation_id = int(cuda_data.get("correlationId", 0))
        start_ns = int(cuda_data.get("startNs", "0"))
        end_ns = int(cuda_data.get("endNs", "0"))

        kernel_data = cuda_data.get("kernel", {})
        kernel_name_id = str(kernel_data.get("demangledName", ""))
        short_name_id = str(kernel_data.get("shortName", ""))

        is_memcpy = False
        memcpy_type = None
        memcpy_size = None

        if event_type == TYPE_CUDA_80:
            is_memcpy = True
            memcpy_data = cuda_data.get("memcpy", {})
            copy_kind = int(memcpy_data.get("copyKind", 0))
            memcpy_size = int(memcpy_data.get("sizebytes", 0))
            # copyKind: 1=HtoD, 2=DtoH, 3=DtoD
            if copy_kind == 1:
                memcpy_type = "Memcpy HtoD"
            elif copy_kind == 2:
                memcpy_type = "Memcpy DtoH"
            elif copy_kind == 3:
                memcpy_type = "Memcpy DtoD"
            else:
                memcpy_type = f"Memcpy (kind={copy_kind})"

        device_id = int(cuda_data.get("deviceId", -1))
        stream_id = str(cuda_data.get("streamId", "0"))

        # 检测是否为同步事件 (eventClass=5 或有 sync 字段)
        is_sync = False
        sync_type = None
        event_class = cuda_data.get("eventClass", 0)
        sync_data = cuda_data.get("sync", {})
        if event_class == 5 or sync_data:
            is_sync = True
            sync_type = sync_data.get("syncType", "SYNC") if sync_data else "SYNC"

        if correlation_id in self.cuda_events_by_sig_corr[sig]:
            existing = self.cuda_events_by_sig_corr[sig][correlation_id]
            if existing.gpu_start_ns is not None and correlation_id != 0:
                self.log_bug(f"Duplicate CudaEvent for correlation_id={correlation_id}, sig={sig}")
            else:
                existing.gpu_start_ns = start_ns
                existing.gpu_end_ns = end_ns
                existing.gpu_duration_ns = end_ns - start_ns
                existing.device_id = device_id
                existing.stream_id = stream_id
                existing.global_pid = global_pid_str
                existing.is_memcpy = is_memcpy or existing.is_memcpy
                existing.is_sync = is_sync or existing.is_sync
                # Update kernel names if not already set
                if not existing.short_name and short_name_id:
                    existing.short_name = short_name_id
                if not existing.kernel_name and kernel_name_id:
                    existing.kernel_name = kernel_name_id
                if memcpy_type:
                    existing.memcpy_type = memcpy_type
                if memcpy_size:
                    existing.memcpy_size_bytes = memcpy_size
                if sync_type:
                    existing.sync_type = sync_type
        else:
            cuda_event = CudaEvent(
                gpu_start_ns=start_ns,
                gpu_end_ns=end_ns,
                gpu_duration_ns=end_ns - start_ns,
                process_signature=sig,
                thread_type=ThreadType.UNKNOWN,
                is_main_thread=False,
                device_id=device_id,
                stream_id=stream_id,
                correlation_id=correlation_id,
                global_pid=global_pid_str,
                kernel_name=kernel_name_id,
                short_name=short_name_id,
                is_memcpy=is_memcpy,
                memcpy_type=memcpy_type,
                memcpy_size_bytes=memcpy_size,
                is_sync=is_sync,
                sync_type=sync_type,
                raw_data=cuda_data
            )
            self.cuda_events_by_sig_corr[sig][correlation_id] = cuda_event

    def build_event_tree(self):
        print("\n" + "=" * 80)
        print("Step 5: Building event tree")
        print("=" * 80)

        # 检查 CudaEvent 是否有 CPU 时间
        events_without_cpu = 0

        for sig, corr_dict in self.cuda_events_by_sig_corr.items():
            for cuda_event in corr_dict.values():
                # 检查是否有 CPU 时间
                if not cuda_event.has_cpu_time():
                    self.log_bug(f"CudaEvent without CPU time: correlation_id={cuda_event.correlation_id}, sig={sig}")
                    events_without_cpu += 1
                    continue

                self.cuda_events_by_signature[sig].append(cuda_event)
                if cuda_event.stream_id and cuda_event.gpu_start_ns is not None:
                    self.cuda_events_by_stream[cuda_event.stream_id].append(cuda_event)

        print(f"Events without CPU time (skipped): {events_without_cpu}")

        for stream_id in self.cuda_events_by_stream:
            self.cuda_events_by_stream[stream_id].sort(key=lambda e: e.gpu_start_ns if e.gpu_start_ns is not None else 0)

        all_signatures = set(self.cuda_events_by_signature.keys()) | set(self.nvtx_by_signature.keys())

        for sig in all_signatures:
            nvtx_list = self.nvtx_by_signature.get(sig, [])
            cuda_list = self.cuda_events_by_signature.get(sig, [])
            all_events = nvtx_list + cuda_list
            # 使用 get_sort_key() 排序
            all_events.sort(key=lambda e: e.get_sort_key())
            self.events_by_signature[sig] = all_events

        print(f"Built event lists for {len(self.events_by_signature)} process signatures")

        total_events = 0
        for sig, events in self.events_by_signature.items():
            total_events += len(events)
            stack = []
            roots = []

            for event in events:
                event.parent = None
                event.children = []

                found_parent = False
                while stack:
                    parent = stack[-1]
                    # 检查时间包含关系
                    if (parent.cpu_start_ns is not None and
                        event.cpu_start_ns is not None and
                        parent.cpu_start_ns <= event.cpu_start_ns and
                        event.cpu_end_ns <= parent.cpu_end_ns):
                        event.parent = parent
                        parent.children.append(event)
                        found_parent = True
                        break
                    else:
                        stack.pop()

                if not found_parent:
                    roots.append(event)

                stack.append(event)

            self.root_events[sig] = roots

        print(f"Total events in trees: {total_events}")

    def analyze_steps(self):
        print("\n" + "=" * 80)
        print("Step 6: Analyzing step GPU execution times")
        print("=" * 80)

        all_iterations = set()
        for nvtx in self.nvtx_events:
            if nvtx.iteration is not None:
                all_iterations.add(nvtx.iteration)

        print(f"Found {len(all_iterations)} unique iterations: {sorted(all_iterations)}")

        for iteration_num in sorted(all_iterations)[:2]:
            print(f"\n{'='*60}")
            print(f"Analyzing iteration {iteration_num}")
            print('='*60)

            # 找到 iteration 的时间范围
            iteration_events = [e for e in self.nvtx_events if e.iteration == iteration_num]
            if not iteration_events:
                continue

            iteration_start = min(e.cpu_start_ns for e in iteration_events)
            iteration_end = max(e.cpu_end_ns for e in iteration_events)

            # 查找该时间范围内的 step 事件
            steps_in_iteration = []
            for nvtx in self.nvtx_events:
                if nvtx.step_type not in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
                    continue
                if iteration_start <= nvtx.cpu_start_ns <= iteration_end:
                    steps_in_iteration.append(nvtx)

            # 按 step_type 分组
            by_type = defaultdict(list)
            for step in steps_in_iteration:
                by_type[step.step_type].append(step)

            print(f"  Found: {len(by_type.get(StepType.FORWARD, []))} forward, "
                  f"{len(by_type.get(StepType.BACKWARD, []))} backward, "
                  f"{len(by_type.get(StepType.OPTIMIZER, []))} optimizer")

            # 分析每种类型的第一个
            for step_type in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
                if step_type not in by_type:
                    continue
                steps = by_type[step_type]
                steps.sort(key=lambda e: e.cpu_start_ns)
                first_step = steps[0]
                self._analyze_step_gpu_time(iteration_num, first_step, step_type)

    def _analyze_step_gpu_time(self, iteration: int, step_nvtx: NvtxEvent, step_type: StepType):
        step_name = step_type.value

        all_cuda_events = []

        def collect_cuda_events(event: BaseEvent):
            if isinstance(event, CudaEvent):
                all_cuda_events.append(event)
            elif isinstance(event, NvtxEvent):
                for child in event.children:
                    collect_cuda_events(child)

        collect_cuda_events(step_nvtx)

        gpu_events = [e for e in all_cuda_events if e.gpu_start_ns is not None]

        if not gpu_events:
            print(f"  iteration {iteration}: {step_name}[{iteration}] (sig={step_nvtx.process_signature}) - No GPU events")
            return

        first_cuda = min(gpu_events, key=lambda e: e.gpu_start_ns)
        last_cuda = max(gpu_events, key=lambda e: e.gpu_end_ns)
        gpu_duration = last_cuda.gpu_end_ns - first_cuda.gpu_start_ns

        print(f"\n  iteration {iteration}: {step_name}[{iteration}] (sig={step_nvtx.process_signature})")
        print(f"    First GPU: {first_cuda.gpu_start_ns} ns")
        print(f"    Last GPU: {last_cuda.gpu_end_ns} ns")
        print(f"    Duration: {gpu_duration} ns ({gpu_duration / 1e6:.2f} ms)")
        print(f"    Total CUDA events: {len(all_cuda_events)}")
        print(f"    GPU events: {len(gpu_events)}")

        execution = StepGPUExecution(
            iteration=iteration,
            step_type=step_type,
            process_signature=step_nvtx.process_signature,
            first_cuda_start_ns=first_cuda.gpu_start_ns,
            last_cuda_end_ns=last_cuda.gpu_end_ns,
            gpu_duration_ns=gpu_duration,
            cuda_events=all_cuda_events,
            nvtx_event=step_nvtx
        )
        self.step_executions.append(execution)

    def analyze_offloading(self):
        print("\n" + "=" * 80)
        print("Step 7: Analyzing FineGrainedOffloading operations")
        print("=" * 80)

        offloading_nvtx = []
        for nvtx in self.nvtx_events:
            if "FineGrainedOffloading" in nvtx.text:
                offloading_nvtx.append(nvtx)

        print(f"Found {len(offloading_nvtx)} FineGrainedOffloading events")

        # 按迭代分组
        by_iteration = defaultdict(list)
        for nvtx in offloading_nvtx:
            iteration = self._infer_iteration(nvtx)
            by_iteration[iteration].append(nvtx)

        for iteration in sorted(by_iteration.keys())[:2]:
            print(f"\n--- Iteration {iteration} ---")
            iteration_events = by_iteration[iteration]

            commit_events = [e for e in iteration_events if "Commit" in e.text]
            start_events = [e for e in iteration_events if "Start" in e.text]

            print(f"  Commit (offload) events: {len(commit_events)}")
            print(f"  Start (reload) events: {len(start_events)}")

            for nvtx in commit_events[:5]:
                self._analyze_offloading_operation(nvtx, "offload", iteration)

            for nvtx in start_events[:5]:
                self._analyze_offloading_operation(nvtx, "reload", iteration)

    def _infer_iteration(self, nvtx: NvtxEvent) -> int:
        nvtx_time = nvtx.cpu_start_ns
        for iteration, events in self.iterations.items():
            for event in events:
                if event.cpu_start_ns <= nvtx_time <= event.cpu_end_ns:
                    return iteration
        return 0

    # 关注的 stream 列表
    TARGET_STREAMS = {'7', '47', '43', '59', '170', '172', '171', '169', '31', '35'}

    def calculate_stream_gpu_time(self):
        """计算每个 NvtxEvent 在指定 stream 上的 GPU 执行时间"""
        print("\n" + "=" * 80)
        print("Calculating stream GPU time for NvtxEvents (target streams only)")
        print("=" * 80)

        # 只统计关注的 stream
        all_streams = set()
        for sig, nvtx_list in self.nvtx_by_signature.items():
            for nvtx in nvtx_list:
                # 收集该 NvtxEvent 下的所有 CudaEvent
                cuda_events = []

                def collect_cuda(evt: BaseEvent):
                    if isinstance(evt, CudaEvent) and evt.stream_id:
                        # 只收集关注的 stream
                        if evt.stream_id in self.TARGET_STREAMS:
                            cuda_events.append(evt)
                            all_streams.add(evt.stream_id)
                    elif isinstance(evt, NvtxEvent):
                        for child in evt.children:
                            collect_cuda(child)

                for child in nvtx.children:
                    collect_cuda(child)

                # 按 stream 分组计算 GPU 时间
                stream_events = defaultdict(list)
                for evt in cuda_events:
                    if evt.gpu_start_ns is not None and not evt.is_sync:
                        stream_events[evt.stream_id].append(evt)

                # 计算每个 stream 上的 GPU 时间范围
                for stream_id, events in stream_events.items():
                    gpu_starts = [e.gpu_start_ns for e in events if e.gpu_start_ns is not None]
                    gpu_ends = [e.gpu_end_ns for e in events if e.gpu_end_ns is not None]
                    if gpu_starts and gpu_ends:
                        nvtx.stream_gpu_time[stream_id] = (
                            min(gpu_starts),
                            max(gpu_ends),
                            max(gpu_ends) - min(gpu_starts)
                        )

        print(f"Found {len(all_streams)} target streams")
        print(f"Streams: {sorted(all_streams)}")

        # 统计有多少 NvtxEvent 有 stream GPU 时间
        nvtx_with_stream_time = sum(1 for sig in self.nvtx_by_signature
                                     for nvtx in self.nvtx_by_signature[sig]
                                     if nvtx.stream_gpu_time)
        print(f"NvtxEvents with stream GPU time: {nvtx_with_stream_time}")

        return sorted(all_streams)

    def build_stream_trees(self, streams: List[str]):
        """为每个 stream 构建树形结构"""
        print("\n" + "=" * 80)
        print("Building stream-level trees")
        print("=" * 80)

        for stream_id in streams:
            # 收集在该 stream 上有 GPU 时间的 NvtxEvent
            nvtx_on_stream = []

            for sig in self.nvtx_by_signature:
                for nvtx in self.nvtx_by_signature[sig]:
                    if stream_id in nvtx.stream_gpu_time:
                        nvtx_on_stream.append(nvtx)

            if not nvtx_on_stream:
                continue

            # 按 stream GPU 开始时间排序
            nvtx_on_stream.sort(key=lambda e: e.stream_gpu_time[stream_id][0])

            # 构建树
            stack = []
            for nvtx in nvtx_on_stream:
                # 初始化该 stream 的 parent/children
                if stream_id not in nvtx.stream_parent:
                    nvtx.stream_parent[stream_id] = None
                if stream_id not in nvtx.stream_children:
                    nvtx.stream_children[stream_id] = []

                # 查找 parent
                while stack:
                    parent = stack[-1]
                    parent_start, parent_end, _ = parent.stream_gpu_time[stream_id]
                    child_start, child_end, _ = nvtx.stream_gpu_time[stream_id]

                    if parent_start <= child_start and child_end <= parent_end:
                        nvtx.stream_parent[stream_id] = parent
                        if stream_id not in parent.stream_children:
                            parent.stream_children[stream_id] = []
                        parent.stream_children[stream_id].append(nvtx)
                        break
                    else:
                        stack.pop()

                stack.append(nvtx)

            print(f"Stream {stream_id}: {len(nvtx_on_stream)} events, built tree")

    def _analyze_offloading_operation(self, parent_nvtx: NvtxEvent, operation_type: str, iteration: int):
        module_desc = parent_nvtx.text
        match = re.search(r'seq\s*=\s*(\d+)', module_desc)
        seq_id = match.group(1) if match else "unknown"

        total_bytes = calculate_tensor_bytes(parent_nvtx.sizes)

        copy_nvtx_list = []
        module_name = ""

        for child in parent_nvtx.children:
            if isinstance(child, NvtxEvent):
                if "aten::copy_" in child.text.lower():
                    copy_nvtx_list.append(child)
                elif "activation" in child.text.lower():
                    parts = child.text.split()
                    if len(parts) >= 3:
                        module_name = parts[-1]

        if not copy_nvtx_list:
            for child in parent_nvtx.children:
                if isinstance(child, NvtxEvent):
                    for grandchild in child.children:
                        if isinstance(grandchild, NvtxEvent) and "aten::copy_" in grandchild.text.lower():
                            copy_nvtx_list.append(grandchild)

        if not copy_nvtx_list:
            return

        memcpy_events = []
        for copy_nvtx in copy_nvtx_list:
            for child in copy_nvtx.children:
                if isinstance(child, CudaEvent) and child.is_memcpy:
                    memcpy_events.append(child)

        total_bytes_from_copies = sum(calculate_tensor_bytes(c.sizes) for c in copy_nvtx_list)
        if total_bytes_from_copies > 0:
            total_bytes = total_bytes_from_copies

        total_memcpy_bytes = 0
        total_memcpy_duration = 0
        first_gpu = None
        last_gpu = None

        for memcpy in memcpy_events:
            if memcpy.gpu_start_ns:
                if first_gpu is None or memcpy.gpu_start_ns < first_gpu:
                    first_gpu = memcpy.gpu_start_ns
                if last_gpu is None or memcpy.gpu_end_ns > last_gpu:
                    last_gpu = memcpy.gpu_end_ns

                if len(memcpy_events) > 0:
                    event_bytes = total_bytes // len(memcpy_events)
                else:
                    event_bytes = 0
                total_memcpy_bytes += event_bytes
                total_memcpy_duration += memcpy.gpu_duration_ns

        throughput = 0
        if total_memcpy_duration > 0:
            throughput = (total_memcpy_bytes / (total_memcpy_duration / 1e9)) / 1e9

        print(f"\n  {operation_type.upper()} - seq={seq_id}, module={module_name}")
        print(f"    Process sig: {parent_nvtx.process_signature}")
        print(f"    Total bytes: {total_bytes / 1e6:.2f} MB")
        print(f"    Copy count: {len(copy_nvtx_list)}")
        print(f"    Memcpy events: {len(memcpy_events)}")
        if first_gpu and last_gpu:
            print(f"    GPU time: {first_gpu} - {last_gpu}")
        print(f"    Throughput: {throughput:.2f} GB/s")

        if memcpy_events and first_gpu and last_gpu:
            is_blocking = self._analyze_copy_blocking(memcpy_events, first_gpu, last_gpu)
        else:
            is_blocking = False

        copy_op = CopyOperation(
            iteration=iteration,
            step_type=parent_nvtx.step_type,
            process_signature=parent_nvtx.process_signature,
            parent_nvtx=parent_nvtx,
            module_name=module_name,
            operation_type=operation_type,
            operation_desc=module_desc,
            copy_count=len(copy_nvtx_list),
            copy_nvtx_events=copy_nvtx_list,
            sizes_per_copy=[e.sizes for e in copy_nvtx_list if e.sizes],
            total_bytes=total_bytes,
            memcpy_events=memcpy_events,
            memcpy_total_bytes=total_memcpy_bytes,
            throughput_gbps=throughput,
            is_blocking=is_blocking,
            gpu_start_ns=first_gpu,
            gpu_end_ns=last_gpu
        )
        self.copy_operations.append(copy_op)

    def _analyze_copy_blocking(self, memcpy_events: List[CudaEvent], copy_start: int, copy_end: int) -> bool:
        if not memcpy_events:
            return False

        copy_stream = memcpy_events[0].stream_id
        concurrent_kernels = []

        for stream_id, events in self.cuda_events_by_stream.items():
            if stream_id == copy_stream:
                continue

            for event in events:
                if event.gpu_start_ns is None or event.is_memcpy:
                    continue

                if (event.gpu_start_ns < copy_end and
                    event.gpu_end_ns > copy_start):
                    concurrent_kernels.append(event)

        has_concurrent = len(concurrent_kernels) > 0

        print(f"    Blocking analysis:")
        print(f"      Concurrent kernels: {len(concurrent_kernels)}")
        print(f"      Is blocking: {not has_concurrent}")

        return not has_concurrent

    def print_bugs(self):
        non_zero_bugs = [b for b in self.bugs if "correlation_id=0" not in b]
        if non_zero_bugs:
            print("\n" + "=" * 80)
            print(f"BUGS FOUND (excluding correlation_id=0): {len(non_zero_bugs)}")
            print("=" * 80)
            for bug in non_zero_bugs[:20]:
                print(f"  - {bug}")
            if len(non_zero_bugs) > 20:
                print(f"  ... and {len(non_zero_bugs) - 20} more")

    def export_iteration_data(self, iteration_num: int, output_dir: str):
        """导出 iteration 16 在 gpu0 上的第一个 forward_step、backward_step、optimizer_step"""
        os.makedirs(output_dir, exist_ok=True)

        # 找到 iteration 的时间范围
        iteration_events = [e for e in self.nvtx_events if e.iteration == iteration_num]
        if not iteration_events:
            print(f"Iteration {iteration_num} not found")
            return

        iteration_start = min(e.cpu_start_ns for e in iteration_events)
        iteration_end = max(e.cpu_end_ns for e in iteration_events)

        print(f"\nExporting gpu0 steps for iteration {iteration_num}")

        # 查找该时间范围内的 step 事件，且主要在 gpu0 上运行
        steps_in_iteration = []
        for nvtx in self.nvtx_events:
            if nvtx.step_type not in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
                continue
            if iteration_start <= nvtx.cpu_start_ns <= iteration_end:
                # 检查是否在 gpu0 上运行
                device_id = self._find_step_device(nvtx)
                if device_id == 0:  # 只保留 gpu0 的事件
                    steps_in_iteration.append((nvtx, device_id))

        # 按 step_type 分组
        by_type = defaultdict(list)
        for nvtx, device_id in steps_in_iteration:
            by_type[nvtx.step_type].append(nvtx)

        print(f"  Found on gpu0: {len(by_type.get(StepType.FORWARD, []))} forward, "
              f"{len(by_type.get(StepType.BACKWARD, []))} backward, "
              f"{len(by_type.get(StepType.OPTIMIZER, []))} optimizer")

        # 找到每种类型的第一个
        for step_type in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
            if step_type not in by_type:
                continue

            steps = by_type[step_type]
            steps.sort(key=lambda e: e.cpu_start_ns)
            first_step = steps[0]

            # 获取 gpu0 的 pcie bus 地址
            device_info = self.devices.get(0)
            pcie_suffix = device_info.pcie_bus.replace(":", "_") if device_info else "unknown"
            self._export_step_data(first_step, output_dir, iteration_num, pcie_suffix)

    def export_stream_trees(self, iteration_num: int, output_dir: str, streams: List[str]):
        """导出指定 iteration 的 step 在每个 stream 上的树（所有 steps，所有 devices）

        输出格式：{output_dir}/{pcie_bus}/{step_name}/stream_{id}.json
        """
        os.makedirs(output_dir, exist_ok=True)

        # 找到 iteration 的时间范围
        iteration_events = [e for e in self.nvtx_events if e.iteration == iteration_num]
        if not iteration_events:
            print(f"Iteration {iteration_num} not found")
            return

        iteration_start = min(e.cpu_start_ns for e in iteration_events)
        iteration_end = max(e.cpu_end_ns for e in iteration_events)

        print(f"\nExporting stream trees for iteration {iteration_num}")

        # 查找该时间范围内的 step 事件，按 device 分组
        steps_by_device: Dict[int, List[NvtxEvent]] = defaultdict(list)
        for nvtx in self.nvtx_events:
            if nvtx.step_type not in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
                continue
            if iteration_start <= nvtx.cpu_start_ns <= iteration_end:
                device_id = self._find_step_device(nvtx)
                if device_id is not None:
                    steps_by_device[device_id].append(nvtx)

        # 为每个 device 导出
        for device_id, steps_in_iteration in steps_by_device.items():
            device_info = self.devices.get(device_id)
            pcie_bus = device_info.pcie_bus if device_info else f"device_{device_id}"
            pcie_suffix = pcie_bus.replace(":", "_")

            # 创建 device 目录
            device_dir = os.path.join(output_dir, pcie_suffix)
            os.makedirs(device_dir, exist_ok=True)

            print(f"\n  Device {device_id} (PCIe: {pcie_bus}): {len(steps_in_iteration)} steps")

            # 按 step_type 分组
            by_type = defaultdict(list)
            for nvtx in steps_in_iteration:
                by_type[nvtx.step_type].append(nvtx)

            # 为每个 step 生成 stream tree
            for step_type in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
                if step_type not in by_type:
                    continue

                steps = by_type[step_type]
                steps.sort(key=lambda e: e.cpu_start_ns)

                print(f"\n    {step_type.value}: {len(steps)} steps")

                # 导出每个 step
                for step_idx, step in enumerate(steps):
                    # 找到该 step 涉及的所有 stream
                    step_streams = [s for s in streams if s in step.stream_gpu_time]

                    if step_idx == 0:
                        print(f"      Step {step_idx}: {len(step_streams)} streams")

                    # 创建 step 目录
                    if step_type == StepType.FORWARD:
                        step_name = f"forward_step[{step_idx}]"
                    elif step_type == StepType.BACKWARD:
                        step_name = f"backward_step[{step_idx}]"
                    else:
                        step_name = "optimizer_step"

                    step_dir = os.path.join(device_dir, step_name)
                    os.makedirs(step_dir, exist_ok=True)

                    for stream_id in step_streams:
                        self._export_stream_tree(step, stream_id, step_dir, step_name)

    def _export_stream_tree(self, step_nvtx: NvtxEvent, stream_id: str, step_dir: str, step_name: str):
        """导出单个 step 在单个 stream 上的树，递归从 GPU 树构建 stream 树

        输出格式：{step_dir}/stream_{stream_id}.json
        """
        filename = os.path.join(step_dir, f"stream_{stream_id}.json")

        def build_stream_tree(evt: BaseEvent, stream_id: str) -> Optional[Dict]:
            """递归构建 stream 树，返回该节点在指定 stream 上的表示

            返回值: dict 或 None（如果该节点及其子孙都不在该 stream 上执行）
            """
            if isinstance(evt, CudaEvent):
                # CudaEvent: 只保留在该 stream 上执行的
                if evt.stream_id != stream_id:
                    return None
                if evt.gpu_start_ns is None:
                    return None
                if evt.is_sync:
                    # 过滤同步事件（如 cudaStreamWaitEvent）
                    return None

                # 获取 TraceProcessEvent 的 name 作为 text
                trace_name = ""
                if evt.raw_data and 'name' in evt.raw_data:
                    name_id = str(evt.raw_data.get('name', ''))
                    if name_id:
                        trace_name = self.resolve_string(name_id)

                # 获取 cuda event 的 short_name（kernel 名称）
                shortname = ""
                if evt.is_memcpy and evt.memcpy_type:
                    shortname = evt.memcpy_type
                elif evt.short_name:
                    shortname = self.resolve_string(evt.short_name)
                elif evt.kernel_name:
                    shortname = self.resolve_string(evt.kernel_name)

                result = {
                    "type": "CudaEvent",
                    "text": trace_name if trace_name else shortname,
                    "shortname": shortname,
                    "stream_id": evt.stream_id,
                    "correlation_id": evt.correlation_id,
                    "stream_start_ns": evt.gpu_start_ns,
                    "stream_end_ns": evt.gpu_end_ns,
                    "stream_duration_ns": evt.gpu_duration_ns,
                    "is_memcpy": evt.is_memcpy,
                }
                if evt.memcpy_type:
                    result["memcpy_type"] = evt.memcpy_type
                if evt.memcpy_size_bytes:
                    result["memcpy_size_bytes"] = evt.memcpy_size_bytes
                return result

            elif isinstance(evt, NvtxEvent):
                # NvtxEvent: 递归收集子节点在该 stream 上的执行
                stream_children = []

                for child in evt.children:
                    child_result = build_stream_tree(child, stream_id)
                    if child_result is not None:
                        stream_children.append(child_result)

                if stream_children:
                    # 有子节点在该 stream 上执行，计算该节点的 stream 时间
                    child_starts = [c.get("stream_start_ns") for c in stream_children if c.get("stream_start_ns") is not None]
                    child_ends = [c.get("stream_end_ns") for c in stream_children if c.get("stream_end_ns") is not None]

                    if child_starts and child_ends:
                        stream_start = min(child_starts)
                        stream_end = max(child_ends)
                        stream_duration = stream_end - stream_start
                    else:
                        stream_start = None
                        stream_end = None
                        stream_duration = 0

                    # 对子节点按 stream 开始时间排序
                    stream_children.sort(key=lambda x: x.get("stream_start_ns", 0) if x.get("stream_start_ns") is not None else 0)

                    return {
                        "type": "NvtxEvent",
                        "text": evt.text,
                        "stream_start_ns": stream_start,
                        "stream_end_ns": stream_end,
                        "stream_duration_ns": stream_duration,
                        "stream_children": stream_children,
                    }
                return None

            return None

        # 构建 stream 树
        result = build_stream_tree(step_nvtx, stream_id)

        if result is None:
            return

        # 构建输出（根节点包装在列表中）
        output_data = [result]

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

        # 统计信息
        def count_events(node: Dict) -> Tuple[int, int]:
            """统计 NvtxEvent 和 CudaEvent 数量"""
            nvtx_count = 0
            cuda_count = 0
            if node.get("type") == "NvtxEvent":
                nvtx_count = 1
                for child in node.get("stream_children", []):
                    nc, cc = count_events(child)
                    nvtx_count += nc
                    cuda_count += cc
            elif node.get("type") == "CudaEvent":
                cuda_count = 1
            return nvtx_count, cuda_count

        nvtx_count, cuda_count = count_events(result)
        print(f"      Exported {step_name} stream {stream_id}: {nvtx_count} NvtxEvents, {cuda_count} CudaEvents")

    def _find_step_device(self, step_nvtx: NvtxEvent) -> Optional[int]:
        """Find the main device for a step based on its process signature"""
        # Use pre-built sig_to_device mapping
        return self.sig_to_device.get(step_nvtx.process_signature)

    def _export_step_data(self, step_nvtx: NvtxEvent, output_dir: str, iteration: int, pcie_suffix: str):
        """导出单个 step 的数据"""
        step_name = step_nvtx.step_type.value
        filename = os.path.join(output_dir, f"iteration_{iteration}_{step_name}_{pcie_suffix}_v3.json")

        def collect_events(event: BaseEvent, depth: int = 0) -> Dict:
            event_data = {
                "type": "NvtxEvent" if isinstance(event, NvtxEvent) else "CudaEvent",
                "cpu_start_ns": event.cpu_start_ns,
                "cpu_end_ns": event.cpu_end_ns,
                "cpu_duration_ns": event.cpu_duration_ns,
                "gpu_start_ns": event.gpu_start_ns,
                "gpu_end_ns": event.gpu_end_ns,
                "gpu_duration_ns": event.gpu_duration_ns,
                "process_signature": event.process_signature,
                "depth": depth,
            }

            if isinstance(event, NvtxEvent):
                event_data["text"] = event.text
                event_data["iteration"] = event.iteration
                event_data["step_type"] = event.step_type.value
                event_data["nvtx_type"] = event.nvtx_type
                event_data["sizes"] = event.sizes
            elif isinstance(event, CudaEvent):
                event_data["kernel_name"] = self.resolve_string(event.kernel_name or "")
                event_data["correlation_id"] = event.correlation_id
                event_data["stream_id"] = event.stream_id
                event_data["device_id"] = event.device_id
                event_data["is_memcpy"] = event.is_memcpy
                event_data["memcpy_type"] = event.memcpy_type

            event_data["children"] = []
            for child in event.children:
                event_data["children"].append(collect_events(child, depth + 1))

            return event_data

        root_data = collect_events(step_nvtx, 0)

        with open(filename, 'w') as f:
            json.dump(root_data, f, indent=2)

        print(f"  Exported {step_nvtx.step_type.value} (pcie={pcie_suffix}) to {filename}")

    def export_step_gpu_excel_v2(self, iteration_num: int, output_dir: str):
        """Export step GPU execution times per stream to Excel (v2 format)

        Output: {output_dir}/{pcie_bus}/summary.xlsx

        Format (same as example4):
        - Row 1: Title
        - Row 2: Stream names (GPU Total, Stream 7, Stream 47...), merged 3 cells each
        - Row 3: Field names (Step Name, start_ns, end_ns, duration_ns...)
        - Row 4+: Data rows for each step
        """
        if not OPENPYXL_AVAILABLE:
            print("openpyxl not available, skipping Excel export")
            return

        # Find iteration events
        iteration_events = [e for e in self.nvtx_events if e.iteration == iteration_num]
        if not iteration_events:
            print(f"Iteration {iteration_num} not found for Excel export")
            return

        iteration_start = min(e.cpu_start_ns for e in iteration_events)
        iteration_end = max(e.cpu_end_ns for e in iteration_events)

        # Collect step events per device
        steps_by_device: Dict[int, List[NvtxEvent]] = defaultdict(list)

        for nvtx in self.nvtx_events:
            if nvtx.step_type not in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
                continue
            if iteration_start <= nvtx.cpu_start_ns <= iteration_end:
                device_id = self._find_step_device(nvtx)
                if device_id is not None:
                    steps_by_device[device_id].append(nvtx)

        # Create one Excel file per device in {pcie_bus}/summary.xlsx
        for device_id, steps in steps_by_device.items():
            device_info = self.devices.get(device_id)
            pcie_bus = device_info.pcie_bus if device_info else f"device_{device_id}"
            pcie_suffix = pcie_bus.replace(":", "_")

            # Create device directory
            device_dir = os.path.join(output_dir, pcie_suffix)
            os.makedirs(device_dir, exist_ok=True)

            filename = os.path.join(device_dir, "summary.xlsx")
            wb = Workbook()
            ws = wb.active
            ws.title = "GPU Times"

            # Sort steps by CPU start time to get order of appearance
            steps.sort(key=lambda e: e.cpu_start_ns)

            # Collect all unique streams across all steps (only target streams)
            all_streams = set()
            step_stream_times = {}  # id(step) -> {stream_id -> (start, end, duration)}

            for step in steps:
                step_streams = {}

                # Get per-stream times from stream_gpu_time (only target streams)
                stream_starts = []
                stream_ends = []
                for stream_id, (gpu_start, gpu_end, gpu_duration) in step.stream_gpu_time.items():
                    if gpu_start is not None and stream_id in self.TARGET_STREAMS:
                        step_streams[stream_id] = (gpu_start, gpu_end, gpu_duration)
                        all_streams.add(stream_id)
                        stream_starts.append(gpu_start)
                        stream_ends.append(gpu_end)

                # GPU Total = earliest start and latest end across all target streams
                if stream_starts and stream_ends:
                    total_start = min(stream_starts)
                    total_end = max(stream_ends)
                    total_duration = total_end - total_start
                    step_streams['total'] = (total_start, total_end, total_duration)
                else:
                    step_streams['total'] = (None, None, None)

                step_stream_times[id(step)] = step_streams

            # Sort streams: 7, 47, 43 first, then others sorted by numeric value
            priority_streams = ['7', '47', '43']
            other_streams = sorted([s for s in all_streams if s not in priority_streams],
                                  key=lambda x: int(x) if x.isdigit() else float('inf'))
            sorted_streams = priority_streams + other_streams

            # Row 1: Title
            num_cols = 1 + (len(sorted_streams) + 1) * 3  # Step Name + (Total + streams) * 3
            ws['A1'] = f"GPU Execution Times - Device {pcie_bus}"
            ws['A1'].font = Font(bold=True, size=14)
            end_col_letter = self._get_col_letter(num_cols)
            ws.merge_cells(f'A1:{end_col_letter}1')

            # Row 2: Stream headers (merged across 3 cells each)
            col = 2  # Start from column B (A is for step name)

            # GPU Total header
            ws.cell(row=2, column=col, value="GPU Total")
            ws.cell(row=2, column=col).font = Font(bold=True)
            ws.cell(row=2, column=col).fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
            ws.cell(row=2, column=col).font = Font(bold=True, color='FFFFFF')
            ws.merge_cells(start_row=2, start_column=col, end_row=2, end_column=col+2)
            col += 3

            # Stream headers
            for stream_id in sorted_streams:
                ws.cell(row=2, column=col, value=f"Stream {stream_id}")
                ws.cell(row=2, column=col).font = Font(bold=True)
                color = '70AD47' if stream_id in priority_streams else 'FFC000'
                ws.cell(row=2, column=col).fill = PatternFill(start_color=color, end_color=color, fill_type='solid')
                ws.cell(row=2, column=col).font = Font(bold=True, color='FFFFFF')
                ws.merge_cells(start_row=2, start_column=col, end_row=2, end_column=col+2)
                col += 3

            # Row 3: Field headers
            col = 1
            ws.cell(row=3, column=col, value="Step Name")
            ws.cell(row=3, column=col).font = Font(bold=True)
            ws.cell(row=3, column=col).fill = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
            col += 1

            for _ in range(len(sorted_streams) + 1):  # +1 for total
                for field_name in ['start_ns', 'end_ns', 'duration_ns']:
                    cell = ws.cell(row=3, column=col, value=field_name)
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color='E7E6E6', end_color='E7E6E6', fill_type='solid')
                    col += 1

            # Data rows
            row = 4
            for step in steps:
                # Use simplified step name
                step_name = simplify_step_name(step.text, step.step_type) if step.text else step.step_type.value
                ws.cell(row=row, column=1, value=step_name)

                col = 2
                stream_times = step_stream_times.get(id(step), {})

                # GPU Total
                total_time = stream_times.get('total', (None, None, None))
                if total_time[0] is not None:
                    ws.cell(row=row, column=col, value=total_time[0])
                    ws.cell(row=row, column=col+1, value=total_time[1])
                    ws.cell(row=row, column=col+2, value=total_time[2])
                col += 3

                # Per-stream times
                for stream_id in sorted_streams:
                    stream_time = stream_times.get(stream_id, (None, None, None))
                    if stream_time[0] is not None:
                        ws.cell(row=row, column=col, value=stream_time[0])
                        ws.cell(row=row, column=col+1, value=stream_time[1])
                        ws.cell(row=row, column=col+2, value=stream_time[2])
                    col += 3

                row += 1

            # Auto-adjust column widths
            ws.column_dimensions['A'].width = 20
            for c in range(2, num_cols + 1):
                col_letter = self._get_col_letter(c)
                ws.column_dimensions[col_letter].width = 15

            wb.save(filename)
            print(f"  Exported summary.xlsx for device {device_id} (PCIe={pcie_bus}) to {filename}")

    def _get_col_letter(self, col_idx: int) -> str:
        """Convert column index to Excel column letter (1=A, 2=B, 27=AA, etc.)"""
        result = ""
        while col_idx > 0:
            col_idx, remainder = divmod(col_idx - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def _get_step_total_gpu_time(self, step_nvtx: NvtxEvent) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Calculate total GPU execution time for a step"""
        gpu_events = []

        def collect_cuda(evt: BaseEvent):
            if isinstance(evt, CudaEvent) and evt.gpu_start_ns is not None:
                gpu_events.append(evt)
            elif isinstance(evt, NvtxEvent):
                for child in evt.children:
                    collect_cuda(child)

        for child in step_nvtx.children:
            collect_cuda(child)

        if gpu_events:
            first_gpu = min(gpu_events, key=lambda e: e.gpu_start_ns)
            last_gpu = max(gpu_events, key=lambda e: e.gpu_end_ns)
            return (first_gpu.gpu_start_ns, last_gpu.gpu_end_ns,
                    last_gpu.gpu_end_ns - first_gpu.gpu_start_ns)
        return (None, None, None)

    def _create_summary_sheet(self, ws, by_type: Dict, device_id: int, pcie_bus: str):
        """Create summary sheet with overall GPU times"""
        # Header
        ws['A1'] = f"GPU Step Execution Summary - Device {device_id} (PCIe: {pcie_bus})"
        ws['A1'].font = Font(bold=True, size=14)
        ws.merge_cells('A1:F1')

        # Column headers
        headers = ['Step Name', 'Process Sig', 'GPU Start (ns)', 'GPU End (ns)', 'Duration (ms)', 'Streams Count']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
            cell.font = Font(bold=True, color='FFFFFF')

        row = 4
        for step_type in [StepType.FORWARD, StepType.BACKWARD, StepType.OPTIMIZER]:
            if step_type not in by_type:
                continue

            steps = by_type[step_type]
            steps.sort(key=lambda e: e.cpu_start_ns)

            for step in steps:
                # Calculate GPU execution time
                gpu_events = []

                def collect_cuda(evt: BaseEvent):
                    if isinstance(evt, CudaEvent):
                        gpu_events.append(evt)
                    elif isinstance(evt, NvtxEvent):
                        for child in evt.children:
                            collect_cuda(child)

                for child in step.children:
                    collect_cuda(child)

                gpu_events = [e for e in gpu_events if e.gpu_start_ns is not None]

                if gpu_events:
                    first_gpu = min(gpu_events, key=lambda e: e.gpu_start_ns)
                    last_gpu = max(gpu_events, key=lambda e: e.gpu_end_ns)
                    gpu_start = first_gpu.gpu_start_ns
                    gpu_end = last_gpu.gpu_end_ns
                    gpu_duration_ms = (gpu_end - gpu_start) / 1e6

                    # Count unique streams
                    streams = set(e.stream_id for e in gpu_events if e.stream_id)

                    # Use the full NVTX text as step name
                    step_name = step.text if step.text else step_type.value
                    ws.cell(row=row, column=1, value=step_name)
                    ws.cell(row=row, column=2, value=step.process_signature)
                    ws.cell(row=row, column=3, value=gpu_start)
                    ws.cell(row=row, column=4, value=gpu_end)
                    ws.cell(row=row, column=5, value=round(gpu_duration_ms, 2))
                    ws.cell(row=row, column=6, value=len(streams))
                    row += 1

        # Auto-adjust column widths
        for col in range(1, 7):
            ws.column_dimensions[chr(64 + col)].width = 18

    def _create_step_stream_sheet(self, ws, steps: List[NvtxEvent], step_type: StepType, device_id: int):
        """Create a sheet showing per-stream GPU execution times for steps"""
        # Header
        ws['A1'] = f"{step_type.value} - Per-Stream GPU Execution Times (Device {device_id})"
        ws['A1'].font = Font(bold=True, size=12)
        ws.merge_cells('A1:G1')

        row = 3
        for step_idx, step in enumerate(steps):
            # Step header - use full NVTX text
            step_display_name = step.text if step.text else f"Step {step_idx + 1}: {step_type.value}"
            ws.cell(row=row, column=1, value=step_display_name)
            ws.cell(row=row, column=1).font = Font(bold=True)
            ws.merge_cells(f'A{row}:G{row}')
            row += 1

            # Collect stream GPU times for this step
            stream_times = []

            for stream_id, (gpu_start, gpu_end, gpu_duration) in step.stream_gpu_time.items():
                if gpu_start is not None:
                    stream_times.append({
                        'stream_id': stream_id,
                        'gpu_start_ns': gpu_start,
                        'gpu_end_ns': gpu_end,
                        'gpu_duration_ns': gpu_duration,
                        'gpu_duration_ms': gpu_duration / 1e6
                    })

            if stream_times:
                # Sort by GPU start time
                stream_times.sort(key=lambda x: x['gpu_start_ns'])

                # Column headers
                headers = ['Stream ID', 'GPU Start (ns)', 'GPU End (ns)', 'Duration (ms)', '% of Total']
                for col, header in enumerate(headers, 1):
                    cell = ws.cell(row=row, column=col, value=header)
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color='70AD47', end_color='70AD47', fill_type='solid')
                    cell.font = Font(bold=True, color='FFFFFF')
                row += 1

                # Calculate total duration for percentage
                total_duration = sum(st['gpu_duration_ns'] for st in stream_times)

                for st in stream_times:
                    percentage = (st['gpu_duration_ns'] / total_duration * 100) if total_duration > 0 else 0

                    ws.cell(row=row, column=1, value=st['stream_id'])
                    ws.cell(row=row, column=2, value=st['gpu_start_ns'])
                    ws.cell(row=row, column=3, value=st['gpu_end_ns'])
                    ws.cell(row=row, column=4, value=round(st['gpu_duration_ms'], 2))
                    ws.cell(row=row, column=5, value=round(percentage, 2))
                    row += 1

                # Total row
                total_ms = total_duration / 1e6
                ws.cell(row=row, column=1, value="Total")
                ws.cell(row=row, column=1).font = Font(bold=True)
                ws.cell(row=row, column=4, value=round(total_ms, 2))
                ws.cell(row=row, column=4).font = Font(bold=True)
                ws.cell(row=row, column=5, value=100.0)
                ws.cell(row=row, column=5).font = Font(bold=True)
                row += 2
            else:
                ws.cell(row=row, column=1, value="No GPU events found")
                row += 2

        # Auto-adjust column widths
        ws.column_dimensions['A'].width = 80  # Wider column for full step names
        ws.column_dimensions['B'].width = 18
        ws.column_dimensions['C'].width = 18
        ws.column_dimensions['D'].width = 15
        ws.column_dimensions['E'].width = 12

    def print_summary(self):
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        summary = defaultdict(lambda: defaultdict(list))
        for exec_info in self.step_executions:
            summary[exec_info.iteration][exec_info.step_type.value].append(exec_info.gpu_duration_ns / 1e6)

        print("\nStep GPU Execution Times:")
        print("-" * 60)
        for iteration in sorted(summary.keys()):
            print(f"\n  Iteration {iteration}:")
            for step_type in ['forward_step', 'backward_step', 'optimizer_step']:
                if step_type in summary[iteration]:
                    durations = summary[iteration][step_type]
                    avg_duration = sum(durations) / len(durations)
                    print(f"    {step_type}: {avg_duration:.2f} ms (avg across {len(durations)} processes)")

        print("\nCopy Operations Summary:")
        print("-" * 60)
        offload_ops = [op for op in self.copy_operations if op.operation_type == "offload"]
        reload_ops = [op for op in self.copy_operations if op.operation_type == "reload"]
        blocking_ops = [op for op in self.copy_operations if op.is_blocking]

        print(f"  Total offload operations: {len(offload_ops)}")
        print(f"  Total reload operations: {len(reload_ops)}")
        print(f"  Blocking operations: {len(blocking_ops)}")

        by_module = defaultdict(lambda: {"offload": 0, "reload": 0, "total_bytes": 0})
        for op in self.copy_operations:
            by_module[op.module_name][op.operation_type] += 1
            by_module[op.module_name]["total_bytes"] += op.total_bytes

        print("\n  By module:")
        for module, stats in by_module.items():
            if module:
                print(f"    {module}: offload={stats['offload']}, reload={stats['reload']}, "
                      f"total_bytes={stats['total_bytes']/1e9:.2f} GB")

    def run(self):
        self.load_devices_from_sqlite()
        self.load_and_parse()
        self.build_event_tree()

        # Build sig_to_device mapping after event tree is built
        self._build_sig_to_device_mapping()

        # 计算 stream GPU 时间并构建 stream 树
        streams = self.calculate_stream_gpu_time()
        self.build_stream_trees(streams)

        self.analyze_steps()
        self.analyze_offloading()
        self.print_bugs()
        self.print_summary()

        iteration = self.iteration

        print("\n" + "=" * 80)
        print(f"Exporting iteration {iteration} stream trees (all devices)")
        print("=" * 80)
        self.export_stream_trees(iteration, self.output_dir, streams)

        print("\n" + "=" * 80)
        print(f"Exporting iteration {iteration} summary.xlsx (all devices)")
        print("=" * 80)
        self.export_step_gpu_excel_v2(iteration, self.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="NSYS Profile Analyzer - Analyze Megatron-LM training GPU performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python nsys_profile_analyzer.py --sqlite profile.sqlite --json profile.json --output ./output
    python nsys_profile_analyzer.py -s profile.sqlite -j profile.json -o ./output -i 16
        """
    )

    parser.add_argument('--sqlite', '-s', required=True,
                        help='Path to NSYS SQLite database file')
    parser.add_argument('--json', '-j', required=True,
                        help='Path to NSYS JSON export file')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for analysis results')
    parser.add_argument('--iteration', '-i', type=int, default=16,
                        help='Iteration number to analyze (default: 16)')

    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.sqlite):
        print(f"Error: SQLite file not found: {args.sqlite}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"NSYS Profile Analyzer V3")
    print(f"=" * 80)
    print(f"SQLite file: {args.sqlite}")
    print(f"JSON file: {args.json}")
    print(f"Output directory: {args.output}")
    print(f"Iteration to analyze: {args.iteration}")
    print(f"=" * 80)

    analyzer = NSYSAnalyzer(args.json, args.sqlite, args.output, args.iteration)
    analyzer.run()

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
