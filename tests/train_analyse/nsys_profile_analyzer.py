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
from dataclasses import dataclass, field, fields
from typing import Optional, List, Dict, Set, Tuple, Any
from enum import Enum
from copy import deepcopy

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

STRING_TABLE = {}

# ============== 常量定义 ==============

TYPE_STRING = "String"
TYPE_TRACE = 47                 # cuda API
TYPE_TRACE_KERNEL = 48          # cuda kernel API, 通过 correlationId 关联到 cuda event, 
TYPE_NVTX_RANGE = 59            # nvtx range event
TYPE_CUDA_KERNEL = 79           # cude kernel event
TYPE_CUDA_MEMCPY = 80           # cuda memcpy event, 记录 memcpy 详细信息
TYPE_CUDA_SYNC_SRC = 106        # cuda sync event, 记录同步源 stream id 和 syncType, 通过 {"eventId":xxx,"eventSyncId":"xxx"} 关联到 type 127
TYPE_CUDA_SYNC_DST = 127        # cuda sync event, 记录同步目的 stream id 和 device id
# TYPE_NVTX_75 = 75           # NVTX 点事件


@dataclass
class DeviceInfo:
    """GPU 设备信息"""
    name: str                   # device name, e.g., "NVIDIA GB200"
    deviceId: int               # CUDA device id
    pcieBus: str = ""           # PCIe bus location, e.g., "0009:01:00.0"
    signatures: List[int] = field(default_factory=list)


@dataclass
class TimeRange:
    startNs: int
    endNs: int
    durationNs: int = field(init=False)

    def __post_init__(self):
        self.durationNs = self.endNs - self.startNs


@dataclass
class Base:
    @classmethod
    def from_dict(cls, data: dict, **kwargs: Any):
        data.update(kwargs)
        cleaned = {
            f.name: f.type(data[f.name])
            for f in fields(cls)
            if f.init and f.name in data and data[f.name] is not None
        }
        return cls(**cleaned)
    

@dataclass
class BaseEvent(Base):
    type: int
    startNs: int
    endNs: int
    durationNs: int = field(init=False)

    def __post_init__(self):
        self.startNs = int(self.startNs)
        self.endNs = int(self.endNs)
        self.durationNs = self.endNs - self.startNs

@dataclass
class NvtxEvent(BaseEvent):
    text: str
    globalTid: int

    @classmethod
    def from_dict(cls, data: dict, **kwargs: Any):
        data['text'] = data.get('Text', '') 
        data['startNs'] = data['Timestamp']
        data['endNs'] = data['EndTimestamp']
        data['globalTid'] = data['GlobalTid']
        return super().from_dict(data, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        self.signature = (self.globalTid >> 24) & 0xFFFFFF 
        self.thread = "main" if (self.globalTid & 0xFFFFFF) == self.signature else "autograd"

@dataclass
class TraceProcessEvent(BaseEvent):
    correlationId: int = 0
    eventClass: int = 0
    name: str = ""
    globalTid: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.name = STRING_TABLE.get(int(self.name), "failed to lookup")
        self.signature = (self.globalTid >> 24) & 0xFFFFFF 
        self.thread = "main" if (self.globalTid & 0xFFFFFF) == self.signature else "autograd"

@dataclass
class CudaEvent(BaseEvent):
    correlationId: int
    deviceId: int
    contextId: int
    streamId: int
    eventClass: int
    globalPid: int
    greenContextId: int = 0
    signature: int = field(init=False)
    kernel: Optional['KernelEvent'] = None
    memcpy: Optional['MemcpyEvent'] = None
    sync: Optional['SyncSrcEvent'] = None
    cudaEventRecord: Optional['SyncDstEvent'] = None
    isSync: bool = False  # 是否为同步事件 (cudaStreamWaitEvent 等)

    @classmethod
    def from_dict(cls, data: dict, **kwargs: Any):
        data.update(kwargs)
        cleaned = {
            f.name: f.type(data[f.name]) if f.name not in ['kernel', 'memcpy', 'sync', 'cudaEventRecord'] else data[f.name]
            for f in fields(cls)
            if f.init and f.name in data and data[f.name] is not None
        }
        return cls(**cleaned)

    def __post_init__(self):
        super().__post_init__()
        self.signature = (self.globalPid >> 24) & 0xFFFFFF

        if self.kernel is not None:
            self.kernel = KernelEvent.from_dict(self.kernel)
        elif self.memcpy is not None:
            self.memcpy = MemcpyEvent.from_dict(self.memcpy)
        elif self.sync is not None:
            self.sync = SyncSrcEvent.from_dict(self.sync)
            self.isSync = True  # 标记为同步事件
        elif self.cudaEventRecord is not None:
            self.cudaEventRecord = SyncDstEvent.from_dict(self.cudaEventRecord)

@dataclass
class KernelEvent(Base):
    """Type:79"""
    shortName: str
    
    def __post_init__(self):
        self.shortName = STRING_TABLE.get(int(self.shortName), "failed to lookup")

@dataclass
class MemcpyEvent(Base):
    """Type:80"""
    sizebytes: int
    copyKind: int 
    srcKind: int
    dstKind: int
    copyCount: int

@dataclass
class SyncSrcEvent(Base):
    """Type:106"""
    eventId: int
    eventSyncId: int
    syncType: str

@dataclass
class SyncDstEvent(Base):
    """Type:127"""
    eventId: int
    eventSyncId: int

    def __post_init__(self):
        self.eventId = int(self.eventId)
        self.eventSyncId = int(self.eventSyncId)


class BaseNode:
    def __init__(self, event: NvtxEvent | TraceProcessEvent):
        self.event = event
        self.cpu_time = TimeRange(event.startNs, event.endNs)
        self.gpu_time: Optional[TimeRange] = None
        self.timeline: Dict[int, Optional[TimeRange]] = {}
        self.children: List['BaseNode'] = []
        self.parent: Optional['BaseNode'] = None


class NvtxNode(BaseNode):
    def __init__(self, event: NvtxEvent):
        super().__init__(event)
        self.name = event.text
        self.fast_parent: 'NvtxNode' = None
        self.fast_children: List['NvtxNode'] = []
        # Stream-level tree structure: {stream_id: List[NvtxNode]}
        self.stream_children: Dict[int, List['NvtxNode']] = {}

        self.analyse_fastpath()

    def search_for_fastpath(self, target: str):
        """
        通过 fastpath查找指定 iteration 下的 target 节点
        """
        print(self.name)
        if getattr(self, self.fastpath) == target:
            return self
        else:
            for child in self.fast_children:
                result = child.search_for_fastpath(target)
                if result is not None:
                    return result
        return None
    
    def analyse_fastpath(self):
        if self.name.startswith("iteration "):
            match = re.search(r'iteration (\d+)', self.name)
            if match:
                self.fastpath = "iteration"
                self.iteration = int(match.group(1))
                return

        if self.name.endswith('optimizer.step'):
            self.fastpath = "step"
            self.step = 'optimizer'
            return

        match = re.search(r'(forward_step\[\d+\])', self.name)
        if match:
            self.fastpath = "step"
            self.step = match.group(1)
            return

        match = re.search(r'(backward_step\[\d+\])', self.name)
        if match:
            self.fastpath = "step"
            self.step = match.group(1)
            return

        if self.name.startswith('activation offloading'):
            self.fastpath = "fine_offload"
            self.fine_offload = self.name.split(' ')[-1]
            return

        if self.name.startswith('activation reloading'):
            self.fastpath = "fine_reload"
            self.fine_reload = self.name.split(' ')[-1]
            return
        
        self.fastpath = None

    def set_fastpath(self, parent = None):
        if parent is not None and self.fastpath is not None:
            self.fast_parent = parent
            parent.fast_children.append(self)
        
        for child in self.children:
            if isinstance(child, NvtxNode):
                child.set_fastpath(self if self.fastpath is not None else parent)

    def compute_stream_timeline(self) -> Dict[int, Optional[TimeRange]]:
        for child in self.children:
            timelines = child.compute_stream_timeline()
            for stream, range in timelines.items():
                if self.timeline.get(stream, None) is None:
                    self.timeline[stream] = TimeRange(range.startNs, range.endNs)
                else:
                    self.timeline[stream].startNs = min(self.timeline[stream].startNs, range.startNs)
                    self.timeline[stream].endNs = max(self.timeline[stream].endNs, range.endNs)
                    self.timeline[stream].durationNs = self.timeline[stream].endNs - self.timeline[stream].startNs
        for stream, timeline in self.timeline.items():
            if self.gpu_time is not None:
                self.gpu_time.startNs = min(self.gpu_time.startNs, timeline.startNs)
                self.gpu_time.endNs = max(self.gpu_time.endNs, timeline.endNs)
                self.gpu_time.durationNs = self.gpu_time.endNs - self.gpu_time.startNs
            else:
                self.gpu_time = TimeRange(timeline.startNs, timeline.endNs)
        return self.timeline
    

    def analyse_step_gpu_time(self):
        if self.fastpath == 'iteration':
            result = {}
            for child in self.fast_children:
                name, timeline = child.analyse_step_gpu_time()
                if name is not None:
                    result[name] = timeline
            return result
        elif self.fastpath == 'step':
            # 基于过滤后的数据计算 timeline
            # 遍历所有子节点，收集非 sync 的 CudaNode
            filtered_timeline = {}

            def collect_cuda_timeline(node: BaseNode):
                if isinstance(node, CudaNode):
                    # 过滤同步事件
                    if not node.isSync:
                        for stream_id, time_range in node.timeline.items():
                            if time_range:
                                if stream_id not in filtered_timeline:
                                    filtered_timeline[stream_id] = []
                                filtered_timeline[stream_id].append(time_range)
                elif isinstance(node, NvtxNode):
                    for child in node.children:
                        collect_cuda_timeline(child)

            # 收集所有子节点的 timeline
            for child in self.children:
                collect_cuda_timeline(child)

            # 合并每个 stream 的时间范围
            result_timeline = {}
            for stream_id, time_ranges in filtered_timeline.items():
                if time_ranges:
                    starts = [tr.startNs for tr in time_ranges]
                    ends = [tr.endNs for tr in time_ranges]
                    result_timeline[stream_id] = TimeRange(min(starts), max(ends))

            return self.step, result_timeline
        return None, None
    
    def analyse_fine_grained_offloading(self):
        """分析细粒度 offload 操作，返回 copy 操作列表"""
        assert self.fastpath.startswith('fine_')

        def _extract_sizes_from_text(text: str) -> List[List[int]]:
            """Extract sizes array from text, handling nested arrays properly."""
            match = re.search(r'sizes\s*=\s*(\[[\[\]\d,\s]+\])', text)
            if match:
                try:
                    sizes_str = match.group(1)
                    return json.loads(sizes_str.replace("'", '"'))
                except:
                    pass
            return []

        # 获取 group name (activation offloading/reloading 后的最后一个词)
        self.group_name = self.name.split()[-1] if self.name else "unknown"

        # 收集所有 copy_ 子节点
        result = []
        for child in self.children:
            if isinstance(child, NvtxNode) and 'aten::copy_' in child.name:
                sizes = _extract_sizes_from_text(child.name)
                # 计算总元素数 - 取第一个非空尺寸
                first_shape = None
                for s in sizes:
                    if s:
                        first_shape = s
                        break

                # 计算元素数：对所有维度求积
                numel = 1
                if first_shape:
                    for dim in first_shape:
                        numel *= dim

                # 查找 copy_ 子节点的 CudaNode
                for c in child.children:
                    if isinstance(c, CudaNode) and c.timeline:
                        # 获取 memcpy 信息 - 需要从 cudaEvent 获取
                        cuda_event = getattr(c.event, 'cudaEvent', None)
                        if cuda_event and hasattr(cuda_event, 'memcpy') and cuda_event.memcpy:
                            memcpy = cuda_event.memcpy
                            time_range = list(c.timeline.values())[0] if c.timeline else None
                            if time_range:
                                duration_ms = time_range.durationNs / 1e6
                                throughput_gib = (memcpy.sizebytes / (1024**3)) / (time_range.durationNs / 1e9) if time_range.durationNs > 0 else 0
                                # 格式化 shape
                                shape_str = str(first_shape) if first_shape else str(sizes)
                                # byte_per_element
                                byte_per_ele = memcpy.sizebytes / numel if numel > 0 else 0

                                result.append({
                                    'shape': shape_str,
                                    'byte_per_element': byte_per_ele,
                                    'size': memcpy.sizebytes,
                                    'time_ms': round(duration_ms, 3),
                                    'throughput_gib_s': round(throughput_gib, 4),
                                })

        self.memcpys = result
        return result


class CudaNode(BaseNode):
    def __init__(self, event: TraceProcessEvent):
        super().__init__(event)
        self.name = event.name
        self.isSync = False  # 是否为同步事件

        if getattr(event, 'cudaEvent', None) is not None:
            cuda_event : CudaEvent = event.cudaEvent
            self.isSync = cuda_event.isSync  # 从 CudaEvent 获取 isSync
            # 过滤同步事件，不添加到 timeline
            if not self.isSync and cuda_event.startNs != 0 and cuda_event.endNs != 0:
                self.gpu_time = TimeRange(cuda_event.startNs, cuda_event.endNs)
                self.timeline[cuda_event.streamId] = self.gpu_time
    
    def compute_stream_timeline(self) -> Dict[int, Optional[TimeRange]]:
        return self.timeline
    
    def analyse_step_gpu_time(self, **kwargs):
        return None
    
    def analyse_fastpath(self):
        self.fastpath = None

# ============== 主分析器类 ==============

class NSYSAnalyzer:
    def __init__(self, json_filepath: str, sqlite_filepath: str, output_dir: str, 
                 ranks: List[int], iteration: int = 16, detail: bool = False):
        self.json_filepath = json_filepath
        self.sqlite_filepath = sqlite_filepath
        self.output_dir = output_dir
        self.target_ranks = ranks
        self.target_iteration = iteration
        self.detail = detail  # 是否导出详细的 step json
        self.devices: Dict[int, DeviceInfo] = {}  # key: cuda device id
        self.sign_to_device: Dict[int, int] = {}  # key: process signature, value: cuda device id

        self.nvtx_events_by_signature: Dict[int, List[NvtxEvent]] = defaultdict(list)
        self.cuda_events_by_signature: Dict[int, List[CudaEvent]] = defaultdict(list)
        self.trace_events_by_signature: Dict[int, List[TraceProcessEvent]] = defaultdict(list)

        self.root_nodes : Dict[int, List[CudaNode|NvtxNode]] = {}       # device
        self.iterations : Dict[int, Dict[int, List[NvtxNode]]] = {}     # divice, iteration

    def load_devices_from_sqlite(self):
        """ step1: 从 sqlite 数据库加载 GPU 设备信息 """
        print("\n")
        print("=" * 80)
        print("Step 1: Loading device info from SQLite")
        print("=" * 80)

        conn = sqlite3.connect(self.sqlite_filepath)
        cursor = conn.cursor()

        # 从 TARGET_INFO_GPU 表读取设备信息
        cursor.execute('SELECT name, busLocation, cuDevice FROM TARGET_INFO_GPU')
        rows = cursor.fetchall()

        for row in rows:
            name, bus_location, cu_device = row
            device_info = DeviceInfo(
                name=name,
                deviceId=cu_device,      # CUDA device id
                pcieBus=bus_location     # e.g., "0009:01:00.0"
            )
            self.devices[cu_device] = device_info
        
        for key in sorted(self.devices.keys()):
            print(f"  Device {self.devices[key].deviceId}: {self.devices[key].name}, PCIe={self.devices[key].pcieBus}")

        conn.close()


    def load_and_parse(self):
        """ step2: 加载 JSON 文件并解析事件 """
        print("\n")
        print("=" * 80)
        print("Step 2: Loading and parsing events")
        print("=" * 80)

        line_count = 0
        nvtx_count = 0
        cuda_count = 0
        trace_count = 0

        global STRING_TABLE

        with open(self.json_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    event_type = record.get('Type', None) if record.get('type', None) == None else record.get('type')

                    if event_type == TYPE_STRING:
                        STRING_TABLE[int(record['id'])] = record['value']
                    elif event_type == TYPE_NVTX_RANGE:
                        event = NvtxEvent.from_dict(record['NvtxEvent'], type=event_type)
                        self.nvtx_events_by_signature[event.signature].append(event)
                        nvtx_count += 1
                    elif event_type in (TYPE_TRACE, TYPE_TRACE_KERNEL):
                        event = TraceProcessEvent.from_dict(record['TraceProcessEvent'], type=event_type)
                        self.trace_events_by_signature[event.signature].append(event)
                        trace_count += 1
                    elif event_type in (TYPE_CUDA_KERNEL, TYPE_CUDA_MEMCPY, TYPE_CUDA_SYNC_SRC, TYPE_CUDA_SYNC_DST):
                        event = CudaEvent.from_dict(record['CudaEvent'], type=event_type)
                        self.cuda_events_by_signature[event.signature].append(event)
                        cuda_count += 1

                        if event_type != TYPE_CUDA_SYNC_SRC:
                            if self.sign_to_device.get(event.signature, None) is None:
                                self.sign_to_device[event.signature] = event.deviceId
                            else:
                                assert self.sign_to_device[event.signature] == event.deviceId

                except json.JSONDecodeError:
                    pass

                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count} lines...")

        for sig, device in self.sign_to_device.items():
            self.devices[device].signatures.append(sig)

        print(f"\nTotal lines: {line_count}")
        print(f"NVTX events: {nvtx_count}")
        print(f"CUDA API events: {trace_count}")
        print(f"CUDA kernel events: {cuda_count}")
        print(f"Device's signatures:")
        for key in sorted(self.devices.keys()):
            print(f"  Device {self.devices[key].deviceId}: {self.devices[key].name}, PCIe={self.devices[key].pcieBus}, signatures={[f'0x{value:06x}' for value in self.devices[key].signatures]}")


    def build_event_tree(self):
        print("\n" + "=" * 80)
        print("Step 3: Building event tree")
        print("=" * 80)

        for rank in self.target_ranks:
            self.iterations[rank] = {}
            # 1. 将 cuda event 和 trace event 匹配
            cpu_events : List[TraceProcessEvent|NvtxEvent] = []
            gpu_events : List[CudaEvent] = []

            maps : Dict[int, CudaEvent] = {}
            for sign, device in self.sign_to_device.items():
                if device == rank:
                    cpu_events.extend(self.trace_events_by_signature.get(sign, []))
                    gpu_events.extend(self.cuda_events_by_signature.get(sign, []))
            
            match = 0
            detach = 0
            for cpu_event in cpu_events:
                maps[cpu_event.correlationId] = cpu_event
            for gpu_event in gpu_events:
                if getattr(gpu_event, 'correlationId', None) is not None:
                    if maps.get(gpu_event.correlationId, None) is not None:
                        match += 1
                        cpu_event = maps[gpu_event.correlationId]
                        assert getattr(cpu_event, 'cudaEvent', None) == None, \
                            f"Error: correlationId {gpu_event.correlationId} of {gpu_event} " \
                            f"already matched with another cudaEvent {getattr(cpu_event, 'cudaEvent')}"
                        # 把 CudaEvent 附加到 TraceProcessEvent 上，这样 CudaNode 可以访问
                        setattr(cpu_event, 'cudaEvent', gpu_event)
                    else:
                        detach += 1
            print(f"device {rank}: matched {match} TraceProcessEvent with CudaEvent, detached {detach} CudaEvent without TraceProcessEvent")

            # 2. 按时间顺序排序
            for sign, device in self.sign_to_device.items():
                if device == rank:
                    cpu_events.extend(self.nvtx_events_by_signature.get(sign, []))
            cpu_events = sorted(cpu_events, key=lambda e: e.startNs)

            # 3. 构建 NvtxNode 和 CudaNode
            nodes = []
            for event in cpu_events:
                if isinstance(event, NvtxEvent):
                    node = NvtxNode(event)
                    nodes.append(node)
                    if node.fastpath == 'iteration':
                        self.iterations[rank][node.iteration] = node
                else:
                    nodes.append(CudaNode(event))

            # 4. 构建 event tree
            total_events = 0
            stack : List[CudaNode|NvtxNode] = []
            roots : List[CudaNode|NvtxNode] = []

            for node in nodes:
                node.parent = None
                node.children = []

                found_parent = False
                while stack:
                    parent = stack[-1]
                    # 检查时间包含关系
                    if (parent.cpu_time.startNs <= node.cpu_time.startNs and \
                        parent.cpu_time.endNs >= node.cpu_time.endNs):
                        node.parent = parent
                        parent.children.append(node)
                        found_parent = True
                        break
                    else:
                        stack.pop()

                if not found_parent:
                    roots.append(node)

                stack.append(node)

            self.root_nodes[rank] = roots

            for i, iteration in self.iterations[rank].items():
                iteration.set_fastpath()

            print(f"device {rank}: built event tree with {len(roots)} root nodes and total {total_events} events")
            for i, root in enumerate(roots):
                    print(f"  root[{i}].name = {root.name} ")


    def compute_stream_timeline(self):
        print("\n" + "=" * 80)
        print("Step 4: Computing stream timelines")
        print("=" * 80)

        for rank in self.target_ranks:
            print(f"device {rank}: computing stream timelines...")
            for node in self.root_nodes[rank]:
                node.compute_stream_timeline()


    def analyze_steps(self):
        print("\n" + "=" * 80)
        print("Step 5: Analyzing step GPU execution times")
        print("=" * 80)

        for rank in self.target_ranks:
            for it, node in self.iterations[rank].items():
                if it == self.target_iteration:
                    steps_data = node.analyse_step_gpu_time()
                    print(f"Found {len(steps_data)} steps")
                    for step_name, timeline in steps_data.items():
                        print(f"Step {step_name}: {len(timeline)} streams")

                    # 生成 Excel 表格，传入 rank 以获取 pcie_bus
                    self._generate_step_analyse_xlsx(steps_data, rank)

    def _generate_step_analyse_xlsx(self, steps_data: dict, rank: int):
        """生成 step_analyse.xlsx 表格"""
        if not OPENPYXL_AVAILABLE:
            print("Warning: openpyxl not available, skipping Excel generation")
            return

        # 获取 pcie_bus，并将冒号替换为下划线
        device_info = self.devices.get(rank)
        if device_info and device_info.pcieBus:
            pcie_bus = device_info.pcieBus.replace(":", "_")
        else:
            pcie_bus = f"device_{rank}"

        # 确定输出路径: {output_dir}/{pcie_bus}/step_analyse.xlsx
        output_path = os.path.join(self.output_dir, pcie_bus, "step_analyse.xlsx")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 收集所有 stream IDs
        all_streams = set()
        for step_name, timeline in steps_data.items():
            all_streams.update(timeline.keys())

        # 排序: 7, 47, 43 放在最前面，其余按大小排序
        priority_streams = [7, 47, 43]
        remaining_streams = sorted([s for s in all_streams if s not in priority_streams])
        ordered_streams = priority_streams + remaining_streams

        # 创建工作簿
        wb = Workbook()
        ws = wb.active
        ws.title = "Step Analysis"

        # GPU Total: 深蓝色 4472C4
        gpu_total_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        # Priority streams: 绿色 70AD47
        priority_stream_fill = PatternFill(start_color='70AD47', end_color='70AD47', fill_type='solid')
        # Other streams: 橙色 FFC000
        other_stream_fill = PatternFill(start_color='FFC000', end_color='FFC000', fill_type='solid')
        # Field headers: 浅灰色 E7E6E6
        field_fill = PatternFill(start_color='E7E6E6', end_color='E7E6E6', fill_type='solid')
        # Step name header
        step_name_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')

        header_font = Font(bold=True, color='FFFFFF')
        field_font = Font(bold=True)
        center_align = Alignment(horizontal='center', vertical='center')

        # 计算总列数
        num_cols = 1 + (len(ordered_streams) + 1) * 3

        # Row 1: 标题 (合并单元格)
        end_col_letter = self._get_col_letter(num_cols)
        ws['A1'] = "GPU Execution Times"
        ws['A1'].font = Font(bold=True, size=14)
        ws.merge_cells(f'A1:{end_col_letter}1')

        # Row 2: Stream headers (每个占3列，合并)
        # 第一列: Step Name
        ws.cell(row=2, column=1, value="")
        ws.cell(row=2, column=1).font = header_font
        ws.cell(row=2, column=1).fill = step_name_fill
        ws.cell(row=2, column=1).alignment = center_align
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=1)

        col = 2
        # GPU Total header
        ws.cell(row=2, column=col, value="GPU Total")
        ws.cell(row=2, column=col).font = header_font
        ws.cell(row=2, column=col).fill = gpu_total_fill
        ws.cell(row=2, column=col).alignment = center_align
        ws.merge_cells(start_row=2, start_column=col, end_row=2, end_column=col+2)
        col += 3

        # Stream headers
        for stream_id in ordered_streams:
            ws.cell(row=2, column=col, value=f"Stream {stream_id}")
            ws.cell(row=2, column=col).font = header_font
            # Priority streams 用绿色，其他用橙色
            if stream_id in priority_streams:
                ws.cell(row=2, column=col).fill = priority_stream_fill
            else:
                ws.cell(row=2, column=col).fill = other_stream_fill
            ws.cell(row=2, column=col).alignment = center_align
            ws.merge_cells(start_row=2, start_column=col, end_row=2, end_column=col+2)
            col += 3

        # Row 3: Field headers (start_ns, end_ns, duration_ns)
        col = 1
        ws.cell(row=3, column=col, value="Step Name")
        ws.cell(row=3, column=col).font = field_font
        ws.cell(row=3, column=col).fill = field_fill
        ws.cell(row=3, column=col).alignment = center_align
        col += 1

        for _ in range(len(ordered_streams) + 1):  # +1 for GPU Total
            for field_name in ['start_ns', 'end_ns', 'duration_ns']:
                cell = ws.cell(row=3, column=col, value=field_name)
                cell.font = field_font
                cell.fill = field_fill
                cell.alignment = center_align
                col += 1

        # 数据行 (从第4行开始)
        row = 4
        for step_name in steps_data.keys():
            timeline = steps_data[step_name]

            # step name
            ws.cell(row=row, column=1, value=step_name)
            ws.cell(row=row, column=1).alignment = center_align

            # 计算 GPU Total (所有 streams 的并集时间范围)
            if timeline:
                all_start = min(tr.startNs for tr in timeline.values())
                all_end = max(tr.endNs for tr in timeline.values())
                all_duration = all_end - all_start
            else:
                all_start = all_end = all_duration = 0

            col = 2
            # GPU Total 数据
            ws.cell(row=row, column=col, value=all_start)
            ws.cell(row=row, column=col + 1, value=all_end)
            ws.cell(row=row, column=col + 2, value=all_duration)
            for i in range(3):
                ws.cell(row=row, column=col + i).alignment = center_align
            col += 3

            # 各个 Stream 数据
            for stream_id in ordered_streams:
                if stream_id in timeline:
                    tr = timeline[stream_id]
                    ws.cell(row=row, column=col, value=tr.startNs)
                    ws.cell(row=row, column=col + 1, value=tr.endNs)
                    ws.cell(row=row, column=col + 2, value=tr.durationNs)
                else:
                    ws.cell(row=row, column=col, value="")
                    ws.cell(row=row, column=col + 1, value="")
                    ws.cell(row=row, column=col + 2, value="")
                for i in range(3):
                    ws.cell(row=row, column=col + i).alignment = center_align
                col += 3

            row += 1

        # 调整列宽 - 参考 old.py 的格式
        ws.column_dimensions['A'].width = 20
        for c in range(2, num_cols + 1):
            col_letter = self._get_col_letter(c)
            ws.column_dimensions[col_letter].width = 15

        # 保存文件
        wb.save(output_path)
        print(f"\nStep analysis saved to: {output_path}")

    def _get_col_letter(self, col_idx: int) -> str:
        """Convert column index to Excel column letter (1=A, 2=B, 27=AA, etc.)"""
        result = ""
        while col_idx > 0:
            col_idx, remainder = divmod(col_idx - 1, 26)
            result = chr(65 + remainder) + result
        return result
                    
        

    def analyze_fine_grained_offloading(self):
        print("\n" + "=" * 80)
        print("Step 6: Analyzing fine-grained offloading")
        print("=" * 80)

        def _collect_fine_grained_nodes(node: NvtxNode, offload_type: str, result: Dict[str, List[NvtxNode]]):
            """递归收集所有 activation offloading/reloading 节点，按 group 名称分组"""
            if not node:
                return
            for n in node.fast_children:
                if n.fastpath and n.fastpath.startswith(f'fine_{offload_type}'):
                    group_name = n.name.split()[-1] if n.name else "unknown"
                    if group_name not in result:
                        result[group_name] = []
                    result[group_name].append(n)
                    continue
                else:
                    _collect_fine_grained_nodes(n, offload_type, result)

        for rank in self.target_ranks:
            fb = self.iterations[rank][self.target_iteration].search_for_fastpath('forward_step[0]')
            bb = self.iterations[rank][self.target_iteration].search_for_fastpath('backward_step[0]')

            fg : Dict[str, List[NvtxNode]] = {}
            bg : Dict[str, List[NvtxNode]] = {}

            _collect_fine_grained_nodes(fb, 'offload', fg)
            _collect_fine_grained_nodes(bb, 'reload', bg)

            print(f"\nRank {rank}: Found {sum(len(v) for v in fg.values())} offload nodes ({len(fg)} groups), "
                  f"{sum(len(v) for v in bg.values())} reload nodes ({len(bg)} groups)")

            # 收集数据并计算每个节点的 memcpys
            offload_all_data = {}
            for group_name, nodes in fg.items():
                offload_all_data[group_name] = nodes
                for node in nodes:
                    node.analyse_fine_grained_offloading()

            reload_all_data = {}
            for group_name, nodes in bg.items():
                reload_all_data[group_name] = nodes
                for node in nodes:
                    node.analyse_fine_grained_offloading()

            # 计算汇总统计
            offload_summary = self._calculate_group_summary(offload_all_data)
            reload_summary = self._calculate_group_summary(reload_all_data)

            # 生成 Excel
            self._generate_fine_grained_excel(offload_all_data, reload_all_data, offload_summary, reload_summary, rank)

    def _calculate_group_summary(self, all_data: Dict[str, List[NvtxNode]]) -> Dict[str, Dict]:
        """计算同名 group 的汇总统计"""
        summary = {}
        for group_name, nodes in all_data.items():
            all_memcpy_sizes = []
            all_memcpy_times = []

            for node in nodes:
                if hasattr(node, 'memcpys') and node.memcpys:
                    for memcpy in node.memcpys:
                        all_memcpy_sizes.append(memcpy['size'])
                        all_memcpy_times.append(memcpy['time_ms'])

            if all_memcpy_sizes:
                total_size = sum(all_memcpy_sizes)
                total_time_ms = sum(all_memcpy_times)
                avg_throughput = (total_size / (1024**3)) / (total_time_ms / 1e3) if total_time_ms > 0 else 0

                summary[group_name] = {
                    'count': len(nodes),
                    'total_size': total_size,
                    'total_time_ms': total_time_ms,
                    'avg_throughput': avg_throughput
                }

        return summary


    def _generate_fine_grained_excel(self, offload_all_data: Dict[str, List[NvtxNode]], reload_all_data: Dict[str, List[NvtxNode]], offload_summary: Dict, reload_summary: Dict, rank: int):
        """生成 fine_grained_offload_analyse.xlsx"""
        if not OPENPYXL_AVAILABLE:
            print("Warning: openpyxl not available, skipping Excel generation")
            return

        # 获取 pcie_bus
        device_info = self.devices.get(rank)
        if device_info and device_info.pcieBus:
            pcie_bus = device_info.pcieBus.replace(":", "_")
        else:
            pcie_bus = f"device_{rank}"

        output_path = os.path.join(self.output_dir, pcie_bus, "fine_grained_offload_analyse.xlsx")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        wb = Workbook()

        # 创建 summary sheet（放在第一个）
        ws_summary = wb.active
        ws_summary.title = "summary"
        self._fill_summary_sheet(ws_summary, offload_summary, reload_summary)

        # 创建 offloading sheet
        ws_offload = wb.create_sheet(title="offloading")
        self._fill_fine_grained_detail_sheet(ws_offload, offload_all_data, "Fine-Grained Offloading")

        # 创建 reloading sheet
        ws_reload = wb.create_sheet(title="reloading")
        self._fill_fine_grained_detail_sheet(ws_reload, reload_all_data, "Fine-Grained Reloading")

        wb.save(output_path)
        print(f"\nFine-grained offload analysis saved to: {output_path}")

    def _fill_summary_sheet(self, ws, offload_summary: Dict, reload_summary: Dict):
        """填充 summary sheet - 同名 group 汇总统计"""
        # Header row
        headers = ['type', 'group', 'count', 'total_size', 'total_time(ms)', 'avg_throughput(GiB/s)']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
            cell.font = Font(bold=True, color='FFFFFF')

        row = 2

        # 写入 offload 汇总数据
        offload_start_row = row
        for group_name, stats in offload_summary.items():
            ws.cell(row=row, column=1, value='offload')
            ws.cell(row=row, column=2, value=group_name)
            ws.cell(row=row, column=3, value=stats['count'])
            ws.cell(row=row, column=4, value=stats['total_size'])
            ws.cell(row=row, column=5, value=round(stats['total_time_ms'], 3))
            ws.cell(row=row, column=6, value=round(stats['avg_throughput'], 4))
            row += 1

        # 合并 offload 的 type 列
        if offload_summary and len(offload_summary) > 1:
            ws.merge_cells(start_row=offload_start_row, start_column=1,
                          end_row=offload_start_row + len(offload_summary) - 1, end_column=1)
        if offload_summary:
            ws.cell(row=offload_start_row, column=1).alignment = Alignment(horizontal='left', vertical='center')

        # 写入 reload 汇总数据
        reload_start_row = row
        for group_name, stats in reload_summary.items():
            ws.cell(row=row, column=1, value='reload')
            ws.cell(row=row, column=2, value=group_name)
            ws.cell(row=row, column=3, value=stats['count'])
            ws.cell(row=row, column=4, value=stats['total_size'])
            ws.cell(row=row, column=5, value=round(stats['total_time_ms'], 3))
            ws.cell(row=row, column=6, value=round(stats['avg_throughput'], 4))
            row += 1

        # 合并 reload 的 type 列
        if reload_summary and len(reload_summary) > 1:
            ws.merge_cells(start_row=reload_start_row, start_column=1,
                          end_row=reload_start_row + len(reload_summary) - 1, end_column=1)
        if reload_summary:
            ws.cell(row=reload_start_row, column=1).alignment = Alignment(horizontal='left', vertical='center')

        # 调整列宽
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 10
        ws.column_dimensions['D'].width = 15
        ws.column_dimensions['E'].width = 18
        ws.column_dimensions['F'].width = 22

    def _fill_fine_grained_detail_sheet(self, ws, all_data: Dict[str, List[NvtxNode]], title: str):
        """填充 fine-grained detail sheet - 只显示每个同名 group 的第一组数据"""
        # Header row
        headers = ['group', 'shape', 'byte/element', 'size', 'time(ms)', 'throughput(GiB/s)',
                   'total_size', 'total_time(ms)', 'average_throughput(GiB/s)']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
            cell.font = Font(bold=True, color='FFFFFF')

        row = 2
        for group_name, nodes in all_data.items():
            if not nodes:
                continue

            # 只取第一个节点的数据
            first_node = nodes[0]
            if not hasattr(first_node, 'memcpys') or not first_node.memcpys:
                continue

            memcpys = first_node.memcpys

            # 计算 group 总计（基于第一组数据）
            total_size = sum(m['size'] for m in memcpys)
            total_time_ms = sum(m['time_ms'] for m in memcpys)
            total_throughput = (total_size / (1024**3)) / (total_time_ms / 1e3) if total_time_ms > 0 else 0

            # 写入每个 copy 操作（仅第一组）
            group_start_row = row
            for idx, memcpy in enumerate(memcpys):
                if idx == 0:
                    ws.cell(row=row, column=1, value=group_name)
                ws.cell(row=row, column=2, value=memcpy['shape'])
                ws.cell(row=row, column=3, value=memcpy['byte_per_element'])
                ws.cell(row=row, column=4, value=memcpy['size'])
                ws.cell(row=row, column=5, value=round(memcpy['time_ms'], 3))
                ws.cell(row=row, column=6, value=round(memcpy['throughput_gib_s'], 4))

                if idx == 0:
                    ws.cell(row=row, column=7, value=total_size)
                    ws.cell(row=row, column=8, value=round(total_time_ms, 3))
                    ws.cell(row=row, column=9, value=round(total_throughput, 4))

                row += 1

            # 合并单元格
            if len(memcpys) > 1:
                ws.merge_cells(start_row=group_start_row, start_column=1,
                              end_row=group_start_row + len(memcpys) - 1, end_column=1)
                ws.merge_cells(start_row=group_start_row, start_column=7,
                              end_row=group_start_row + len(memcpys) - 1, end_column=7)
                ws.merge_cells(start_row=group_start_row, start_column=8,
                              end_row=group_start_row + len(memcpys) - 1, end_column=8)
                ws.merge_cells(start_row=group_start_row, start_column=9,
                              end_row=group_start_row + len(memcpys) - 1, end_column=9)
            ws.cell(row=group_start_row, column=1).alignment = Alignment(horizontal='left', vertical='center')

        # 调整列宽
        ws.column_dimensions['A'].width = 20
        for c in range(2, 10):
            ws.column_dimensions[chr(64 + c)].width = 18


    # 关注的 stream 列表
    TARGET_STREAMS = {7, 47, 43, 59, 170, 172, 171, 169, 31, 35}

    def calculate_stream_gpu_time(self):
        """计算每个 NvtxNode 在指定 stream 上的 GPU 执行时间"""
        print("\n" + "=" * 80)
        print("Calculating stream GPU time for NvtxNodes (target streams only)")
        print("=" * 80)

        all_streams = set()

        for rank in self.target_ranks:
            for it, node in self.iterations[rank].items():
                if it == self.target_iteration:
                    # 递归遍历所有节点
                    self._calculate_node_stream_time(node, all_streams)

        print(f"Found {len(all_streams)} target streams")
        print(f"Streams: {sorted(all_streams)}")
        return sorted(all_streams)

    def _calculate_node_stream_time(self, node: NvtxNode, all_streams: Set[int]):
        """递归计算节点的 stream GPU 时间"""
        # 收集该节点下的所有 CudaNode（过滤掉 sync event）
        cuda_nodes = []

        def collect_cuda(n: BaseNode):
            if isinstance(n, CudaNode):
                # 过滤同步事件（如 cudaStreamWaitEvent）
                if not n.isSync:
                    cuda_nodes.append(n)
            elif isinstance(n, NvtxNode):
                for child in n.children:
                    collect_cuda(child)

        for child in node.children:
            collect_cuda(child)

        # 按 stream 分组
        stream_events: Dict[int, List[CudaNode]] = defaultdict(list)
        for cuda_node in cuda_nodes:
            for stream_id, time_range in cuda_node.timeline.items():
                if stream_id in self.TARGET_STREAMS and time_range:
                    stream_events[stream_id].append(cuda_node)
                    all_streams.add(stream_id)

        # 计算每个 stream 上的 GPU 时间范围
        for stream_id, events in stream_events.items():
            gpu_starts = [n.timeline[stream_id].startNs for n in events if stream_id in n.timeline]
            gpu_ends = [n.timeline[stream_id].endNs for n in events if stream_id in n.timeline]
            if gpu_starts and gpu_ends:
                if stream_id not in node.timeline:
                    node.timeline[stream_id] = TimeRange(min(gpu_starts), max(gpu_ends))

        # 递归处理子节点
        for child in node.fast_children:
            self._calculate_node_stream_time(child, all_streams)

    def build_stream_trees(self, streams: List[int]):
        """为每个 stream 构建树形结构"""
        print("\n" + "=" * 80)
        print("Building stream-level trees")
        print("=" * 80)

        for rank in self.target_ranks:
            for it, node in self.iterations[rank].items():
                if it == self.target_iteration:
                    for stream_id in streams:
                        self._build_stream_tree_for_node(node, stream_id)

    def _build_stream_tree_for_node(self, node: NvtxNode, stream_id: int):
        """为单个节点构建单个 stream 的树"""
        # 收集在该 stream 上有 GPU 时间的子节点（过滤掉 sync event）
        children_on_stream = []

        for child in node.children:
            if isinstance(child, CudaNode):
                # 过滤同步事件
                if child.isSync:
                    continue
                if stream_id in child.timeline and child.timeline[stream_id]:
                    children_on_stream.append(child)
            elif isinstance(child, NvtxNode):
                # 递归检查子节点
                self._build_stream_tree_for_node(child, stream_id)
                # 如果子节点或其子孙在该 stream 上有执行，则加入
                if stream_id in child.timeline and child.timeline[stream_id]:
                    children_on_stream.append(child)

        # 按 GPU 开始时间排序
        children_on_stream.sort(key=lambda n: n.timeline[stream_id].startNs if stream_id in n.timeline and n.timeline[stream_id] else 0)

        # 保存为该节点的 stream_children
        if children_on_stream:
            if stream_id not in node.stream_children:
                node.stream_children[stream_id] = []
            node.stream_children[stream_id] = children_on_stream

    def export_stream_trees(self, streams: List[int]):
        """导出指定 iteration 的 step 在每个 stream 上的树"""
        print("\n" + "=" * 80)
        print(f"Exporting stream trees for iteration {self.target_iteration}")
        print("=" * 80)

        for rank in self.target_ranks:
            # 获取 pcie_bus
            device_info = self.devices.get(rank)
            if device_info and device_info.pcieBus:
                pcie_bus = device_info.pcieBus.replace(":", "_")
            else:
                pcie_bus = f"device_{rank}"

            for it, node in self.iterations[rank].items():
                if it == self.target_iteration:
                    # 找到所有 step 节点
                    for child in node.fast_children:
                        if child.fastpath == 'step':
                            step_name = child.step
                            self._export_step_stream_trees(child, pcie_bus, step_name, streams)

    def _export_step_stream_trees(self, step_node: NvtxNode, pcie_bus: str, step_name: str, streams: List[int]):
        """导出单个 step 的所有 stream trees"""
        # 创建目录
        step_dir = os.path.join(self.output_dir, pcie_bus, step_name)
        os.makedirs(step_dir, exist_ok=True)

        for stream_id in streams:
            if stream_id in step_node.stream_children:
                self._export_stream_tree_json(step_node, stream_id, step_dir, step_name)

    def _export_stream_tree_json(self, node: NvtxNode, stream_id: int, step_dir: str, step_name: str):
        """导出单个 stream 的树到 JSON"""
        filename = os.path.join(step_dir, f"stream_{stream_id}.json")

        def build_stream_tree(n: BaseNode) -> Optional[Dict]:
            """递归构建 stream 树"""
            if isinstance(n, CudaNode):
                # CudaNode: 只保留在该 stream 上执行的
                if stream_id not in n.timeline or not n.timeline[stream_id]:
                    return None
                # 过滤同步事件（如 cudaStreamWaitEvent）
                if n.isSync:
                    return None

                tr = n.timeline[stream_id]
                return {
                    "type": "CudaNode",
                    "name": n.name,
                    "stream_id": stream_id,
                    "stream_start_ns": tr.startNs,
                    "stream_end_ns": tr.endNs,
                    "stream_duration_ns": tr.durationNs,
                }
            elif isinstance(n, NvtxNode):
                # NvtxNode: 递归收集子节点在该 stream 上的执行
                stream_children = []

                for child in n.children:
                    child_result = build_stream_tree(child)
                    if child_result is not None:
                        stream_children.append(child_result)

                if stream_children:
                    # 有子节点在该 stream 上执行
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
                        "type": "NvtxNode",
                        "name": n.name,
                        "stream_start_ns": stream_start,
                        "stream_end_ns": stream_end,
                        "stream_duration_ns": stream_duration,
                        "stream_children": stream_children,
                    }
                return None
            return None

        # 构建 stream 树
        result = build_stream_tree(node)

        if result is None:
            return

        # 构建输出（根节点包装在列表中）
        output_data = [result]

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  Exported {step_name} stream {stream_id} to {filename}")


    def run(self):
        self.load_devices_from_sqlite()
        self.load_and_parse()
        self.build_event_tree()
        self.compute_stream_timeline()

        # 计算 stream GPU 时间并构建 stream trees
        streams = self.calculate_stream_gpu_time()
        self.build_stream_trees(streams)

        # 导出 stream trees
        self.export_stream_trees(streams)

        self.analyze_steps()
        self.analyze_fine_grained_offloading()

def main():
    parser = argparse.ArgumentParser(
        description="NSYS Profile Analyzer - Analyze Megatron-LM training GPU performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python nsys_profile_analyzer.py --sqlite profile.sqlite --json profile.json --output ./output
    python nsys_profile_analyzer.py -s profile.sqlite -j profile.json -o ./output -i 16
    python nsys_profile_analyzer.py -s profile.sqlite -j profile.json -o ./output -r 0 1
    python nsys_profile_analyzer.py -s profile.sqlite -j profile.json -o ./output --detail
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
    parser.add_argument('--rank', '-r', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Device ranks to export (default: 0 1 2 3, export all devices)')
    parser.add_argument('--detail', action='store_true',
                        help='Export detailed step JSON files (default: False, only export xlsx)')

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
    print(f"Ranks to export: {args.rank}")
    print(f"Detail mode: {'enabled' if args.detail else 'disabled'}")
    print(f"=" * 80)

    analyzer = NSYSAnalyzer(args.json, args.sqlite, args.output, args.rank, args.iteration, args.detail)
    analyzer.run()

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
