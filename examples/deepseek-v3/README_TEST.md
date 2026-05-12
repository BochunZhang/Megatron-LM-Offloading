# DeepSeek-V3 Test Architecture

## Directory Structure (4-Layer Architecture)

```
examples/deepseek-v3/
├── bin/
│   ├── train.sh          # Layer 1: Execution Engine
│   ├── features.sh       # Layer 2: Feature Controller
│   ├── run-test.sh       # Layer 3: Test Runner
│   └── validate-env.sh   # Environment validation (TODO)
├── testcases/
│   ├── ep4-alltoall.yaml   # EP4+alltoall testcase
│   ├── ep4-hybridep.yaml   # EP4+hybridep testcase
│   └── vpp4-alltoall.yaml  # VPP testcase
├── lib/
│   └── utils.sh          # Shared utility functions (TODO)
└── README_TEST.md        # This file
```

## Usage

### 1. List Available Testcases
```bash
./bin/run-test.sh --list
```

### 2. Execute Single Testcase
```bash
# Default execution
./bin/run-test.sh --test ep4-alltoall

# Enable features
./bin/run-test.sh --test ep4-alltoall --features graph
./bin/run-test.sh --test ep4-alltoall --features graph,profile
./bin/run-test.sh --test ep4-alltoall --feature graph --feature offload_act

# Override mbs parameter
./bin/run-test.sh --test ep4-alltoall --mbs 1,2

# Override training iterations
./bin/run-test.sh --test ep4-alltoall --iters 10

# Dry run (print commands only)
./bin/run-test.sh --test ep4-alltoall --features graph --dry-run
```

### 3. Execute All Testcases
```bash
./bin/run-test.sh --test-all --features graph
```

### 4. Direct Layer 1 Call (Advanced Users)
```bash
./bin/train.sh \
  --tp 1 --pp 1 --ep 4 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --num-layer 5 --num-expert 32 \
  --dispatcher alltoall \
  --enable-cuda-graph
```

## Supported Features

| Feature | Description | Mapped Parameter |
|---------|-------------|------------------|
| graph | Enable CUDA Graph | `--enable-cuda-graph` |
| profile | Enable profiling | `--profile` |
| offload_act | Activation Offloading | `--activation-offload` |
| offload_wt | Weight Offloading | `--weights-offload` |
| offload_opt | Optimizer Offloading | `--optimizer-offload` |
| fine_grained | Fine-grained Offloading | `--fine-grained-offload` |

## Migration Guide

### From Old Scripts

| Old Command | New Command |
|-------------|-------------|
| `./run_testcase.sh --test 5layer-ep4-alltoall --graph` | `./bin/run-test.sh --test ep4-alltoall --features graph` |
| `./run_testcase.sh --test 5layer-ep4-alltoall --graph --offload act` | `./bin/run-test.sh --test ep4-alltoall --features graph,offload_act` |

### Add New Testcase

1. Create YAML file in `testcases/` directory
2. Define configuration following existing file format
3. Verify with `./bin/run-test.sh --list`

## Backup

Original scripts backed up to: `examples/back-deepseek/v2026.05.12/`
