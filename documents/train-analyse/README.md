# NSYS Profile Analysis Documentation

This directory contains the NSYS Profile Analyzer for DeepSeek-V3 training performance analysis.

## Directory Structure

```
documents/train_analyse/
├── design.md                    # Design documentation
├── README.md                      # This file
└── example/                       # Example analysis results
    ├── analysis_report.md         # Analysis report with tables
    ├── iteration_16_gpu_times_*.xlsx      # Excel files (4 devices)
    └── iteration_16_*_step_*.json           # JSON tree files (gpu0)
```

## Design Documentation

See [design.md](design.md) for detailed design information:
- Data structures (BaseEvent, CudaEvent, NvtxEvent)
- TID/PID structure and process signature calculation
- Analysis workflow
- Output format specifications

## Example Analysis Results

### GPU Devices

| Device ID | PCIe Bus | GPU Model |
|-----------|----------|-----------|
| 0 | 0008:01:00.0 | NVIDIA GB200 |
| 1 | 0009:01:00.0 | NVIDIA GB200 |
| 2 | 0018:01:00.0 | NVIDIA GB200 |
| 3 | 0019:01:00.0 | NVIDIA GB200 |

### Analysis Tables

The analysis report (`example/analysis_report.md`) contains 4 independent tables, one for each device:

- **Device 0 (PCIe: 0008:01:00.0)**: 4 forward, 11 backward, 8 optimizer steps
- **Device 1 (PCIe: 0009:01:00.0)**: 4 forward, 1 backward, 8 optimizer steps
- **Device 2 (PCIe: 0018:01:00.0)**: 4 forward, 2 backward, 8 optimizer steps
- **Device 3 (PCIe: 0019:01:00.0)**: 4 forward, 2 backward, 8 optimizer steps

### Table Format

Each table shows:
- **Step Name**: Full NVTX marker name (e.g., `megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining.backward_step[0]`)
- **Total GPU Duration (ms)**: Overall GPU execution time for the step
- **Streams Count**: Number of CUDA streams used
- **Per-Stream Duration (ms)**: GPU execution time on each individual stream

### Example Step Names

| Step Type | Full NVTX Name |
|-----------|----------------|
| forward_step | `megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining.forward_step[0]` |
| backward_step | `megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining.backward_step[0]` |
| optimizer_step | `megatron.training.training.train_step.optimizer.step` |

### Average Step Times (Iteration 16)

| Step Type | Average Duration | Step Count |
|-----------|-----------------|------------|
| Forward Step | 143.77 ms | 16 steps across 4 devices |
| Backward Step | 128.06 ms | 16 steps across 4 devices |
| Optimizer Step | 31.01 ms | 32 steps across 4 devices |

### Generated Files

**Excel Files (per device):**
- `iteration_16_gpu_times_0008_01_00.0.xlsx` (Device 0)
- `iteration_16_gpu_times_0009_01_00.0.xlsx` (Device 1)
- `iteration_16_gpu_times_0018_01_00.0.xlsx` (Device 2)
- `iteration_16_gpu_times_0019_01_00.0.xlsx` (Device 3)

Each Excel file contains:
- Summary worksheet: All steps with GPU start/end times and durations
- forward_step worksheet: Per-stream details for forward steps
- backward_step worksheet: Per-stream details for backward steps
- optimizer_step worksheet: Per-stream details for optimizer steps

**JSON Files (Device 0 only):**

**Step tree files**: Contain complete event hierarchy with NvtxEvent and CudaEvent
- `iteration_16_forward_step_0008_01_00.0_v3.json`
- `iteration_16_backward_step_0008_01_00.0_v3.json`
- `iteration_16_optimizer_step_0008_01_00.0_v3.json`

**Stream tree files**: `iteration_16_*_stream_*_0008_01_00.0_v3.json`

Stream tree structure includes:
- **NvtxEvent**: Python NVTX markers with `stream_children`
- **CudaEvent**: GPU kernel/memcpy events with full details:
  - `correlation_id`: CUDA correlation ID
  - `stream_id`: CUDA stream ID
  - `kernel_name`: Kernel name
  - `is_memcpy`: Whether it's a memory copy operation
  - `memcpy_type`: Type of memory copy (HtoD, DtoH, DtoD)
  - `device_id`: GPU device ID
  - `global_tid`: Global thread ID
  - `global_pid`: Global process ID

Example stream tree structure:
```json
{
  "type": "NvtxEvent",
  "text": "megatron.core.transformer.moe.moe_layer.preprocess.token_dispatcher.dispatch_preprocess",
  "stream_children": [
    {
      "type": "NvtxEvent",
      "text": "aten::copy_",
      "stream_children": [
        {
          "type": "CudaEvent",
          "correlation_id": 5686,
          "stream_id": "31",
          "is_memcpy": true,
          "gpu_start_ns": 193860303,
          "gpu_end_ns": 193863823
        }
      ]
    }
  ]
}
```

## Usage

### Python Analyzer

```bash
python tests/train_analyse/nsys_profile_analyzer.py \
    --sqlite profile.sqlite \
    --json profile.json \
    --output ./results \
    --iteration 16
```

### Shell Script

```bash
./tests/train_analyse/run.sh --iteration 16 logs/nsys-profile/
```

### DeepSeek Training with Comparison Mode

```bash
./examples/deepseek-v3/train_deepseek_v3_gb200.sh \
    --compare-fine-grained-offload \
    --fine-grained-offload
```

## Key Findings

1. **Step Execution Pattern**: Forward and backward steps alternate in chronological order, followed by optimizer steps at the end of each iteration.

2. **Multi-Stream Execution**: Each step uses multiple CUDA streams (6-10 streams), with Stream 7 being the primary compute stream.

3. **Device Variations**: Different devices may execute different numbers of steps due to distributed training load balancing.

4. **Fine-Grained Offloading**: The analysis captures FineGrainedOffloading operations with memory copy events per module (qkv_linear, core_attn, etc.).

## Tools Location

- Analyzer: `tests/train_analyse/nsys_profile_analyzer.py`
- Run script: `tests/train_analyse/run.sh`
- Report generator: `tests/train_analyse/generate_report.py`
