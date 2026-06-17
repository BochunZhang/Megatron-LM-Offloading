# Nsys 文档分析

## 1. deviceId 与 PCIe 的映射关系
sqlite 文件中，`TARGET_INFO_GPU` 定义了 device id 和 PCIe busLocation 的映射关系。
| id | cuDevice | name | busLocation | uuid |
| :----: | :----: | :----: | :----: | :----: |
| 0 | 0 | NVIDIA GB200 | 0008:01:00.0 | f500a7d0-2c78-e74d-c297-6c1484addf52 |
| 1 | 1 | NVIDIA GB200 | 0009:01:00.0 | 2e6d735c-1c20-1003-834c-7de877d9d577 |
| 2 | 2 | NVIDIA GB200 | 0018:01:00.0 | 8864c989-157a-96af-4734-61aa25a93bab |
| 3 | 3 | NVIDIA GB200 | 0019:01:00.0 | 21b34ad1-4187-9cae-b275-b6e05684358f |

## 2. Pid 与 Tid 映射关系

64-bit GlobalTid/GlobalPid 结构:

```
0x0001 0a16ce 0a16ce
│      │       │
│      │       └─ 低 24 bits: 线程标识
│      └─ 中 24 bits: 进程标识
└─ 高 16 bits: 固定前缀 (0x0001)
```

Pid/Tid 是一个 64 bits 的整数，可以分割为 "高 16 bit" / "中 24 bit" / "低 24 bit"，其中 "高 16 bit" 固定为 0x0001，"中 24 bit" 标识不同进程，"低 24 bit" 表示不同线程。
- 主进程的 "低 24 bit" 为 0x000000"，e.g. 0x0001 0a16ce 000000 对应一个进程 0x0a16ce。
- 主线程的 "低 24 bit" 与 "中 24 bit" 相同，e.g. 0x0001 0a16ce 0a16ce 是进程 0a16ce 的主线程。
- autograd: "低 24 bit" 与 "中 24 bit" 不同，e.g. 0x0001 0a16ce xxxxxx 是进程 0a16ce 的子线程，负责 autograd 的计算。


## 3. EventType 分析


| Type | 数量 | 事件类别 | 说明 |
|:----:|:----:|:----:|:----:|
| 27 | 219 | CommEvent | 进程启动命令 |
| 41 | 22,504 | SchedEvent | CPU 调度事件（线程切换） |
| 47 | 5 | TraceProcessEvent | CUDA API 跟踪事件（correlationId=0） |
| 48 | 255,047 | TraceProcessEvent | CUDA API 跟踪事件（主要 API 调用） |
| 49 | 77 | DiagnosticEvent | 诊断/日志事件 |
| 59 | 295,885 | NvtxEvent | NVTX 范围事件（带起止时间） |
| 75 | 12 | NvtxEvent | NVTX 标记事件（单点时间戳） |
| 79 | 50,742 | CudaEvent | CUDA Kernel 执行事件 |
| 80 | 8,268 | CudaEvent | CUDA 内存拷贝事件 (memcpy) |
| 106 | 48,792 | CudaEvent | CUDA 同步事件 |
| 127 | 17,952 | CudaEvent | CUDA Event 记录事件 |


### NvtxEvent 标记 cpu 执行过程
```json
{
    "Type":59,
    "NvtxEvent":{
        "Type":59,
        "Timestamp":"27651616",
        "Text":"iteration 16",
        "GlobalTid":"292568055813839",  // 0x 0001 0a16cf 0a16cf
        "EndTimestamp":"1450326304",
        "DomainId":"0",
        "NsTime":true
    }
}
```
- 记录了 nvtx text / 启动时间 / 结束时间 / tid 等信息.
- "iteration 16" 在主线程上执行，这段代码的在 cpu 上执行时间为 27651616ns ~ 1450326304ns.
- Tid 相同的 NvtxEvent 可以根据 Timestamp & EndTimestamp 确定依赖关系.

### TraceProcessEvent & CudaEvent 标记普通 cuda kernel
```json
{
    "Type":48,
    "TraceProcessEvent":{
        "startNs":"140547168",
        "endNs":"140562944",
        "correlationId":745,
        "eventClass":1,
        "name":"330",                   // cuLaunchKernelEx
        "returnValue32":0,
        "globalTid":"292568055813839"   // 0x0001 0a16cf 0a16cf
    }
}

{
    "Type":79,
    "CudaEvent":{
        "startNs":"140564259",
        "endNs":"140619331",
        "correlationId":745,
        "deviceId":1,
        "contextId":"1",
        "streamId":"7",
        "eventClass":3,
        "globalPid":"292568055152640",  // 0x0001 0a16cf 000000
        "greenContextId":"0",
        "kernel":{
            "demangledName":"316",      // nvjet_qqtst_128x256_128x6_2x2_2cta_h_bz_Avec32UE8M0_Bvec32UE8M0_TNT
            "shortName":"316",
            "mangledName":"316",
            "eventCategory":"316",
            ...
        }
    }
}
```

- device 1 的主线程，在 cuda stream 7 上创建了一个 nvjet kernel，这个 kernel 执行 GEMM 计算
- 主线程在 140547168ns ~ 140562944ns 向 gpu 提交任务
- gpu 在 140564259ns ~ 140619331ns 执行该任务
- 通过 TraceProcessEvent 的 Tid 和 CudaEvent 的 Pid 可以确定两个 event 发生在进程 0a16cf 上
- 通过 correlationId 将二者绑定在一起
- CudaEvent type 为 79 表示通过 launch kernel 启动的 event

### TraceProcessEvent & CudaEvent 标记 Memcpy

```json
{
    "Type":48,
    "TraceProcessEvent":{
        "startNs":"145737344",
        "endNs":"145764320",
        "correlationId":1138,
        "eventClass":0,
        "name":"305",                   // cudaMemcpyAsync_v3020
        "returnValue32":0,
        "globalTid":"292568055813839"   // 0x0001 0a16cf 0a16cf
    }
}
{
    "Type":80,
    "CudaEvent":{
        "startNs":"145769443",
        "endNs":"146102147",
        "correlationId":1138,
        "deviceId":1,
        "contextId":"1",
        "streamId":"43",
        "eventClass":1,
        "globalPid":"292568055152640",  // 0x0001 0a16cf 000000
        "greenContextId":"0",
        "memcpy":{
            "sizebytes":"58720256",
            "copyKind":"2",
            "srcKind":2,
            "dstKind":1,
            "copyCount":"1"
        }
    }
}
```

- device 1 的主线程，在 cuda stream 7 上创建了一个 memcpy 执行 D2H 的数据拷贝
- 通过 Pid / Tid / correlationId 能够将 memcpy 在 cpu 和 gpu 的执行时间对应起来
- CudaEvent 记录了 memcpy 的细节，e.g. 拷贝了 58720256 字节，以及从什么设备拷贝到了什么设备上等信息
- CudaEvent type 为 80 表示通过 cudaMemcpyAsync 启动的内存拷贝

### TraceProcessEvent & CudaEvent 标记 Stream Wait Event

wait event 是让 stream A 等待 stream B 执行完毕后再执行，该操作是非阻塞的.

e.g. fine-grained offload

- forward: compute / wait / offload memcpy
  - offload stream (43) 等待 compute stream (7) 结束后执行 offload memcpy
- backward: wait / wait / reload / wait / backward，
  - reload stream (47) 等待 compute stream (7) 结束
  - reload stream (47) 等待 offload stream (43) 结束后执行 reload memcpy
  - compute stream (7) 等待 reload stream (47) 结束后后执行 backward 计算

```json
{
    "Type":48,
    "TraceProcessEvent":{
        "startNs":"291260768",
        "endNs":"291262720",
        "correlationId":16337,
        "eventClass":0,
        "name":"338",                   // cudaStreamWaitEvent_v3020
        "returnValue32":0,
        "globalTid":"292568055814936"   // 0x0001 0a16cf 0a1b18
    }
}

{
    "Type":106,
    "CudaEvent":{
        "startNs":"291260928",
        "endNs":"291262048",
        "correlationId":16337,
        "deviceId":0,
        "contextId":"1",
        "streamId":"47",
        "eventClass":5,
        "globalPid":"292568055152640",  // 0x0001 0a16cf 000000
        "sync":{
            "eventId":8823,
            "eventSyncId":"205",
            "syncType":"SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT"
        }
    }
}

{
    "Type":127,
    "CudaEvent":{
        "startNs":"0",
        "endNs":"0",
        "correlationId":10596,
        "deviceId":1,
        "contextId":"1",
        "streamId":"43",
        "eventClass":6,
        "globalPid":"292568055152640",  // 0x0001 0a16cf 000000
        "greenContextId":"0",
        "cudaEventRecord":{
            "eventId":8823,
            "eventSyncId":"205"
        }
    }
}

{
    "Type":48,
    "TraceProcessEvent":{
        "startNs":"235275584",
        "endNs":"235278784",
        "correlationId":10596,
        "eventClass":0,
        "name":"298",
        "returnValue32":0,
        "globalTid":"292568055813839"   // 0x0001 0a16cf 0a16cf
    }
}

```

- 该片段是 autograd 的执行过程，stream 47 等待 stream 43
- Stream Wait Event 涉及到 1 个 TraceProcessEvent 和 2 个 CudaEvent
- TraceProcessEvent 记录 cpu 创建 stream event 到 wait 等待完毕的时间
- type=106 的 CudaEvent 记录等待线程 A (stream 47) 的信息，包括等待的时间范围
- type=127 的 CudaEvent 记录阻塞线程 B (stream 43) 的信息
- type=106 记录的 deviceId = 0 是无效信息 (均为 0)，type=127 的 CudaEvent 记录的 deviceId = 1 是有效信息
- type=106 和 type=127 通过 eventId & eventSyncId 一一映射

### TraceProcessEvent & CudaEvent 标记 Stream Sync Event

backward 结束后，新的一轮 forward 需要 sync 后执行。

```json
{
    "Type":48,
    "TraceProcessEvent":{
        "startNs":"397428416",
        "endNs":"401172416",
        "correlationId":33966,
        "eventClass":0,
        "name":"309",                   // cudaStreamSynchronize_v3020
        "returnValue32":0,
        "globalTid":"292568055813839"   // 0x0001 0a16cf 0a16cf
    }
}

{
    "Type":106,
    "CudaEvent":{
        "startNs":"397429376",
        "endNs":"401172192",
        "correlationId":33966,
        "deviceId":0,
        "contextId":"1",
        "streamId":"7",
        "eventClass":5,
        "globalPid":"292568055152640",  // 0x0001 0a16cf 000000
        "sync":{
            "eventId":4294967295,
            "eventSyncId":"4294967295",
            "syncType":"SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT"
        }
    }
}
```

- 该片段是 backward 结束后，forward 等待 sync 的过程
- name=309 对应 cudaStreamSynchronize_v3020，表示这是一个 sync 过程
- 在 default stream 上同步，sync 耗时为 3.744 ms
- 这里仍然错误标注了 deviceId，但通过 Pid 能够识别出其对应 device 1 (0009:01:00.0)


### 

```json
{
    "Type":48,
    "TraceProcessEvent":{
        "startNs":"259075360",
        "endNs":"260812352",
        "correlationId":13096,
        "eventClass":0,
        "name":"522",               // cudaEventSynchronize_v3020
        "returnValue32":0,
        "globalTid":"292568055813839"
    }
}

{
    "Type":106,
    "CudaEvent":{
        "startNs":"259075744",
        "endNs":"260811968",
        "correlationId":13096,
        "deviceId":0,
        "contextId":"1",
        "streamId":"4294967295",
        "eventClass":5,
        "globalPid":"292568055152640",
        "sync":{
            "eventId":30406,
            "eventSyncId":"238",
            "syncType":"SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE"
        }
    }
}

{
    "Type":127,
    "CudaEvent":{
        "startNs":"0",
        "endNs":"0",
        "correlationId":13089,
        "deviceId":1,
        "contextId":"1",
        "streamId":"31",
        "eventClass":6,
        "globalPid":"292568055152640",
        "greenContextId":"0",
        "cudaEventRecord":{
            "eventId":30406,
            "eventSyncId":"238"
        }
    }
}
```

- 该片段是 alltoall ep 的过程中，等待当前 stram 的 event 结束
- name=522 对应 cudaEventSynchronize_v3020
- 在 stream 31 上同步，sync 耗时为 1.737 ms
- type=106 仍然错误标注了 deviceId

代码中使用的常量（对应 nsys_profile_analyzer.py）：

```python
TYPE_TRACE_PROCESS_47 = 47    # TraceProcessEvent (correlationId=0)
TYPE_TRACE_PROCESS_48 = 48    # TraceProcessEvent (主要 API)
TYPE_NVTX_59 = 59             # NvtxEvent (范围事件)
TYPE_NVTX_75 = 75             # NvtxEvent (标记事件)
TYPE_CUDA_79 = 79             # CudaEvent (Kernel)
TYPE_CUDA_80 = 80             # CudaEvent (Memcpy)
TYPE_CUDA_106 = 106           # CudaEvent (同步)
TYPE_CUDA_127 = 127           # CudaEvent (EventRecord)
```