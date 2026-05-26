# DeepSeek-V3 Test Framework

## Directory Structure

```
examples/deepseek-v3/
├── bin/
│   ├── run.sh            # Test runner (orchestrates test execution)
│   └── train.sh          # Training execution engine
├── testcases/
│   ├── ep4-alltoall.yaml     # Testcase: EP4 + alltoall dispatcher
│   ├── ep4-hybridep.yaml     # Testcase: EP4 + hybridep dispatcher
│   └── vpp4-alltoall.yaml    # Testcase: VPP4 + alltoall dispatcher
└── README.md             # This file
```

## Architecture

- **train.sh**: Low-level training execution script
  - Receives expanded parameter names (--pipeline-parallel, --tensor-parallel, etc.)
  - Supports advanced features: --profile, --graph, --offload-act, --offload-weight, --offload-optim, --offload-fine, --offload-fine-modules
  - No testcase logic, only executes training

- **run.sh**: Testcase orchestration script
  - Parses YAML testcase configurations
  - Expands matrix parameters (e.g., mbs: [1,2,4,8] → 4 runs)
  - Converts features to train.sh arguments
  - Manages log directories

## Usage

### 1. List Available Testcases
```bash
./bin/run.sh list
```

### 2. Execute Single Testcase
```bash
# Normal execution
./bin/run.sh test ep4-alltoall

# Dry-run mode (print commands only, do not execute)
./bin/run.sh -d test ep4-alltoall
./bin/run.sh --dry-run test ep4-alltoall
```

### 3. Direct Layer 1 Call (Advanced Users)
```bash
./bin/train.sh \
  --pipeline-parallel 1 \
  --tensor-parallel 1 \
  --expert-parallel 4 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --num-layer 5 \
  --num-expert 32 \
  --dispatcher alltoall \
  --graph
```

## YAML Configuration Format

```yaml
name: testcase-name
description: Test description

param:
  pp: 1                          # pipeline-parallel
  tp: 1                          # tensor-parallel
  ep: 4                          # expert-parallel
  global-batch-size: 128
  num-expert: 32
  num-layer: 5
  moe-freq: "([0]*2+[1]*3)"
  seq-length: 4096
  dispatcher: alltoall
  pp-layout: ""                  # optional: pipeline layout

matrix:
  mbs: [1, 2, 4, 8]              # micro-batch-size variations

feature:
  profile: false                 # enable profiling
  graph: false                   # enable CUDA graph
  offload: []                    # options: [act, weight, optim]
  fine_grained: []               # fine-grained offload modules, e.g. [attn_norm, core_attn, ...]
```

### Parameter Mapping

| YAML Param | train.sh Argument |
|------------|-------------------|
| pp | --pipeline-parallel |
| tp | --tensor-parallel |
| ep | --expert-parallel |
| mbs | --micro-batch-size |
| gbs | --global-batch-size |
| num-expert | --num-expert |
| num-layer | --num-layer |
| moe-freq | --moe-freq |
| seq-length | --seq-length |
| seq-len | --seq-length |
| dispatcher | --dispatcher |
| pp-layout | --pipeline-parallel-layout |

### Feature Mapping

| YAML Feature | train.sh Argument |
|--------------|-------------------|
| profile: true | --profile |
| graph: true | --graph |
| offload: [act] | --offload-act |
| offload: [weight] | --offload-weight |
| offload: [optim] | --offload-optim |
| fine_grained: [module1, ...] | --offload-fine --offload-fine-modules module1 ... |

**Available fine-grained offload modules:** `attn_norm`, `qkv_linear`, `core_attn`, `attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`

When using `fine_grained` without specifying modules, default modules will be used:
`attn_norm`, `qkv_linear`, `core_attn`, `attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`

## Matrix Expansion

The `matrix` section defines parameter sweeps. Each combination is executed sequentially:

```yaml
matrix:
  mbs: [1, 2]
  tp: [1, 2]
```

This generates 4 test runs: (mbs=1,tp=1), (mbs=1,tp=2), (mbs=2,tp=1), (mbs=2,tp=2)

## Log Management

Before running tests:
- Existing `logs/` directory is renamed to `logs-{timestamp}`
- New `logs/` directory is created

After running tests:
- All YAML testcase files are copied to `logs/`
- Training logs are saved in individual subdirectories

## Adding New Testcase

1. Create YAML file in `testcases/` directory
2. Define `param:` section with base configuration
3. Define `matrix:` section for parameter sweeps (optional)
4. Define `feature:` section for advanced features (optional)
5. Verify with `./bin/run.sh list`
