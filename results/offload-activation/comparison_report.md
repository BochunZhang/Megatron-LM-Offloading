# Offload Activation Analysis Report

**Analysis Date:** 2026-05-27

**Test Configuration:**
- Model: DeepSeek-V3
- Config: dp0-tp1-pp1-ep4-mbs2-gbs32-expert32-layer5-seq4096
- Iteration Analyzed: 16

---

## Summary

This report analyzes the **time increase when enabling offload** on two hardware platforms:
- **H100** (NVIDIA L20X, 8 GPUs)
- **GB200** (NVIDIA GB200, 4 GPUs)

---

## Time Increase Analysis: Offload Enable vs Disable

### H100 Platform

| Step | Disable (ms) | Enable (ms) | Increase (ms) | Increase (%) |
|------|--------------|-------------|---------------|--------------|
| **Forward** | 115.71 | 330.54 | **+214.83** | **+185.7%** |
| **Backward** | 229.94 | 269.59 | +39.65 | +17.2% |
| **Optimizer** | 167.25 | 165.49 | -1.76 | -1.1% |
| **Total** | 512.90 | 765.62 | **+252.72** | **+49.3%** |

**H100 Key Findings:**
- Forward step suffers the most: **+214.83 ms** (nearly 3x slower)
- Backward step increases moderately: +39.65 ms
- Optimizer step slightly improved: -1.76 ms
- **Total iteration time increases by 49.3%**

---

### GB200 Platform

| Step | Disable (ms) | Enable (ms) | Increase (ms) | Increase (%) |
|------|--------------|-------------|---------------|--------------|
| **Forward** | 123.36 | 148.78 | **+25.42** | **+20.6%** |
| **Backward** | 112.31 | 119.31 | +7.00 | +6.2% |
| **Optimizer** | 66.01 | 70.42 | +4.41 | +6.7% |
| **Total** | 301.68 | 338.51 | **+36.83** | **+12.2%** |

**GB200 Key Findings:**
- Forward step increase is moderate: **+25.42 ms**
- Backward step increase is minimal: +7.00 ms
- Optimizer step increase is small: +4.41 ms
- **Total iteration time increases by 12.2%**

---

## Cross-Platform Comparison

### Forward Step Increase

| Platform | Time Increase (ms) | Time Increase (%) | Ratio |
|----------|-------------------|-------------------|-------|
| **H100** | +214.83 | +185.7% | **8.5x worse** |
| **GB200** | +25.42 | +20.6% | Baseline |

**Observation:** H100 forward step increase is **8.5x larger** than GB200 in absolute time.

### Backward Step Increase

| Platform | Time Increase (ms) | Time Increase (%) | Ratio |
|----------|-------------------|-------------------|-------|
| **H100** | +39.65 | +17.2% | **5.7x worse** |
| **GB200** | +7.00 | +6.2% | Baseline |

**Observation:** H100 backward step increase is **5.7x larger** than GB200.

### Total Iteration Increase

| Platform | Time Increase (ms) | Time Increase (%) | Impact |
|----------|-------------------|-------------------|--------|
| **H100** | +252.72 | +49.3% | **High** |
| **GB200** | +36.83 | +12.2% | **Moderate** |

---

## Visual Summary

### Time Increase by Step (milliseconds)

```
Forward Step Increase:
H100  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ +214.83 ms
GB200 ████████████ +25.42 ms

Backward Step Increase:
H100  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ +39.65 ms
GB200 ███████ +7.00 ms

Total Iteration Increase:
H100  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ +252.72 ms
GB200 ████████████████████████████████████████████████████████████████████ +36.83 ms
```

---

## Offload Event Analysis

### H100 Offload-Enable
- Offload Events: 1,392
- Reload Events: 1,392
- Total: 2,784 operations
- **Blocking Operations: 0**

### GB200 Offload-Enable
- Offload Events: 1,392
- Reload Events: 1,392
- Total: 2,784 operations
- **Blocking Operations: 1** (qkv_linear on Device 3)

**Note:** Despite having a blocking operation, GB200 handles offload much more efficiently.

---

## Root Cause Analysis

### Why H100 Forward Step Increases 185.7%?

1. **Memory Bandwidth Limitation**
   - Offload requires moving data to system memory
   - H100 PCIe bandwidth is a bottleneck
   - Forward computation stalls waiting for data

2. **Poor Overlap**
   - Offload operations not well overlapped with compute
   - Sequential execution pattern observed

3. **Communication Overhead**
   - All-to-all communication conflicts with offload
   - Stream synchronization issues

### Why GB200 Performs Better?

1. **Higher Memory Bandwidth**
   - Better HBM bandwidth utilization
   - PCIe 5.0 vs H100's PCIe 4.0

2. **Better Overlap**
   - Offload operations concurrent with kernels
   - Non-blocking operations (mostly)

3. **Efficient EP Handling**
   - Expert Parallelism streams (169-172) well managed
   - Less interference with offload streams

---

## Recommendations

### For H100

**Not Recommended to Enable Offload for this workload**
- 49.3% performance degradation is significant
- Forward step becomes 3x slower
- Consider only if:
  - Memory capacity is critical constraint
  - Batch size reduction is not an option
  - Can tolerate 50% throughput loss

### For GB200

**Acceptable to Enable Offload**
- 12.2% degradation is moderate
- Enables larger batch sizes or models
- Memory vs Performance tradeoff is reasonable

**Optimization Opportunity:**
- Address 1 blocking operation on Device 3
- Could potentially reduce total increase to ~10%

---

## Generated Analysis Files

### H100
- `results/offload-activation/h100/offload-disable/analyse/`
  - 4 devices: summary.xlsx + stream trees
- `results/offload-activation/h100/offload-enable/analyse/`
  - 4 devices: summary.xlsx + stream trees

### GB200
- `results/offload-activation/gb200/offload-disable/analyse/`
  - 4 devices: summary.xlsx + stream trees
- `results/offload-activation/gb200/offload-enable/analyse/`
  - 4 devices: summary.xlsx + stream trees

---

## Technical Notes

- Analysis performed using `nsys_profile_analyzer.py`
- Profiles captured on 2026-05-27
- GB200 offload-enable data is the optimized version (formerly enable2)
- Iteration 16 selected for consistent comparison
- Offload modules: qkv_linear, core_attn
