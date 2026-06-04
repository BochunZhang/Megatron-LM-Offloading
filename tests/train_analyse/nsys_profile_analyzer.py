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
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
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
        self.gpu_time = Optional[TimeRange]
        self.timeline: Dict[int, Optional[TimeRange]] = {}
        self.children: List['BaseNode'] = []
        self.parent: Optional['BaseNode'] = None


class NvtxNode(BaseNode):
    def __init__(self, event: NvtxEvent):
        super().__init__(event)
        self.name = event.text
        self.fast_parent: 'NvtxNode' = None
        self.fast_children: List['NvtxNode'] = []

        self.analyse_fastpath()

    def search_for_fastpath(self, target: str):
        """
        通过 fastpath查找指定 iteration 下的 target 节点
        """
        if getattr(self, self.fastpath) == target:
            return self
        else:
            for child in self.fast_children:
                result = child.search_for_fastpath(target)
                if result is not None:
                    return result
        return None
    
    def analyse_type(self):
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

        match = re.search(r'forward_step\[\d+\]', self.name)
        if match:
            self.fastpath = "step"
            self.step = f'forward_step[{match.group(0)}]'
            return

        match = re.search(r'backward_step\[\d+\]', self.name)
        if match:
            self.fastpath = "step"
            self.step = f'backward_step[{match.group(0)}]'
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
        if parent is not None and parent.fastpath is not None:
            self.fast_parent = parent
            parent.fast_children.append(self)
        
        for child in self.children:
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
                name, time = child.analyse_step_gpu_time()
                if name is not None:
                    result[name] = time
            return result
        elif self.fastpath == 'step':
                return self.step, self.gpu_time
        return None, None
    
    def analyse_fine_grained_offloading(self):
        assert self.fastpath.startswith('fine_')

        def _extract_sizes_from_text(text: str) -> List[List[int]]:
            """Extract sizes array from text, handling nested arrays properly."""
            # "aten::copy_, op_id = 15, sizes = [[2, 4096], [2, 4096], []], input_op_ids = [(14,0), (0,-1), (0,-1)]"
            match = re.search(r'sizes\s*=\s*(\[[\[\]\d,\s]+\])', text)
            if match:
                try:
                    sizes_str = match.group(1)
                    return json.loads(sizes_str.replace("'", '"'))
                except:
                    pass
            return []
        
        result: list[MemcpyEvent] = defaultdict()
        for child in self.fast_children:
            if child.name.startswith('aten::copy_'):
                sizes = _extract_sizes_from_text(child.name)
                numel = 1
                for s in sizes:
                    numel *= s[0]
                child.shape = sizes
                
                for c in child.children:
                    if getattr(c.event.cudaEvent, 'memcpy', None) is not None:
                        memcpy = c.event.cudaEvent.memcpy
                        memcpy.numel = numel
                        memcpy.time = TimeRange(c.event.CudaEvent.startNs, c.event.CudaEvent.endNs).durationNs
                        memcpy.bytePerEle = memcpy.sizebytes / numel
                        memcpy.throughput = memcpy.sizebytes / (memcpy.time.durationNs * 1e-9)
                        result.append(memcpy)
        # summary = {
        #     'size': sum([m.sizebytes for m in result]),
        #     'duration': sum([m.time.durationNs for m in result]),
        #     # 'duration': result[-1].time.endNs - result[0].time.startNs
        # }
        # summary['throughput'] = summary['size'] / (summary['duration'] * 1e-9)
        self.memcpys = result


class CudaNode(BaseNode):
    def __init__(self, event: TraceProcessEvent):
        super().__init__(event)
        self.name = event.name

        if getattr(event, 'cudaEvent', None) is not None:
            cuda_event : CudaEvent = event.cudaEvent
            if cuda_event.startNs != 0 and cuda_event.endNs != 0:
                self.gpu_time = TimeRange(cuda_event.startNs, cuda_event.endNs)
                self.timeline[cuda_event.streamId] = self.gpu_time
    
    def compute_stream_timeline(self) -> Dict[int, Optional[TimeRange]]:
        return self.timeline
    
    def analyse_step_gpu_time(self, **kwargs):
        return None

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
                        setattr(gpu_event, 'cudaEvent', cpu_event)
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
                    result = node.analyse_step_gpu_time()
                    print(result['forward_step[0]'])
                    # 制表...
                    
        

    def analyze_fine_grained_offloading(self):
        print("\n" + "=" * 80)
        print("Step 6: Analyzing fine-grained offloading")
        print("=" * 80)

        def _fine_grained_group(node: NvtxNode, offload: str, result: Dict[str, List[NvtxNode]]):
            for n in node.fast_children:
                if n.fastpath == f'fine_{offload}':
                    result[n.fastpath].append(n)
                    return
                else:
                    _fine_grained_group(n, offload, result)
   
        for rank in self.target_ranks:
            fb = self.iterations[rank][self.target_iteration].search_for_fastpath('forward_step[0]')
            bb = self.iterations[rank][self.target_iteration].search_for_fastpath('backward_step[0]')
            fg : Dict[str, List[NvtxNode]] = defaultdict(list)
            bg : Dict[str, List[NvtxNode]] = defaultdict(list)

            _fine_grained_group(fb, 'offload', fg)
            _fine_grained_group(bb, 'reload', bg)

            offload : Dict[str, List[MemcpyEvent]] = defaultdict(list)
            for key, group in fg.items():
                for node in group:
                    node.analyse_fine_grained_offloading()
                offload[key] = group[0].memcpys
            
            offload_summary = {
                key: {
                    'size': sum([m.sizebytes for m in group[0].memcpys]),
                    'duration': sum([m.time.durationNs for m in group[0].memcpys]),
                }
                for key, group in fg.items()
                # 'duration': result[-1].time.endNs - result[0].time.startNs
            }
            for key, dict in offload_summary.items():
                dict[key]['throughput'] = dict['size'] / (dict['duration'] * 1e-9)

            reload : Dict[str, List[MemcpyEvent]] = defaultdict(list)
            for key, group in bg.items():
                for node in group:
                    node.analyse_fine_grained_offloading()
                reload[key] = group[0].memcpys
            reload_summary = {
                key: {
                    'size': sum([m.sizebytes for m in group[0].memcpys]),
                    'duration': sum([m.time.durationNs for m in group[0].memcpys]),
                }
                for key, group in bg.items()
                # 'duration': result[-1].time.endNs - result[0].time.startNs
            }
            for key, dict in reload_summary.items():
                dict[key]['throughput'] = dict['size'] / (dict['duration'] * 1e-9)

            # 制表...


    def run(self):
        self.load_devices_from_sqlite()
        self.load_and_parse()
        self.build_event_tree()
        self.compute_stream_timeline()
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
