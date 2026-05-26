# DeepSeek-V3 测试框架

## 目录结构

```
examples/deepseek-v3/
├── bin/
│   ├── run.sh            # 测试运行脚本（负责编排测试执行）
│   └── train.sh          # 训练执行引擎
├── testcases/
│   ├── ep4-alltoall.yaml     # 测试用例：EP4 + alltoall dispatcher
│   ├── ep4-hybridep.yaml     # 测试用例：EP4 + hybridep dispatcher
│   └── vpp4-alltoall.yaml    # 测试用例：VPP4 + alltoall dispatcher
└── README_zh.md          # 本文件（中文文档）
```

## 架构设计

本测试框架采用分层设计，各脚本职责分离：

### train.sh - 训练执行引擎（第一层）
- 接收展开后的完整参数名（如 `--pipeline-parallel`、`--tensor-parallel` 等）
- 支持高级特性参数：`--profile`、`--graph`、`--offload-act`、`--offload-weight`、`--offload-optim`、`--offload-fine`、`--offload-fine-modules`
- 不包含任何测试用例逻辑，仅负责执行训练任务
- 可直接被高级用户调用，用于单次训练

### run.sh - 测试用例编排脚本（第二层）
- 解析 YAML 格式的测试用例配置
- 展开矩阵参数（如 `mbs: [1,2,4,8]` 会生成 4 次运行）
- 将 YAML 中的 `feature` 配置转换为 train.sh 的命令行参数
- 管理日志目录（自动归档旧日志、创建新日志目录、测试完成后复制 YAML 文件）

## 使用方式

### 1. 列出所有可用测试用例
```bash
./bin/run.sh list
```

### 2. 执行单个测试用例
```bash
# 正常执行
./bin/run.sh test ep4-alltoall

# 仅打印命令而不执行（用于调试或查看将要执行的命令）
./bin/run.sh -d test ep4-alltoall
./bin/run.sh --dry-run test ep4-alltoall
```

### 3. 直接调用底层脚本（高级用户）
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

## YAML 配置格式

```yaml
name: testcase-name
description: 测试描述

param:
  pp: 1                          # pipeline-parallel（流水线并行度）
  tp: 1                          # tensor-parallel（张量并行度）
  ep: 4                          # expert-parallel（专家并行度）
  global-batch-size: 128         # 全局批次大小
  num-expert: 32                 # 专家数量
  num-layer: 5                   # 层数
  moe-freq: "([0]*2+[1]*3)"      # MoE层频率
  seq-length: 4096               # 序列长度
  dispatcher: alltoall           # dispatcher 类型
  pp-layout: ""                  # 可选：流水线并行布局

matrix:
  mbs: [1, 2, 4, 8]              # micro-batch-size 变化组合

feature:
  profile: false                 # 是否启用性能分析
  graph: false                   # 是否启用 CUDA Graph
  offload: []                    # 卸载选项：[act, weight, optim]
  fine_grained: []               # 细粒度卸载模块，如 [attn_norm, core_attn, ...]
```

### 参数映射表

| YAML 参数名 | train.sh 参数名 | 说明 |
|------------|----------------|------|
| pp | --pipeline-parallel | 流水线并行度 |
| tp | --tensor-parallel | 张量并行度 |
| ep | --expert-parallel | 专家并行度 |
| mbs | --micro-batch-size | 微批次大小 |
| gbs | --global-batch-size | 全局批次大小 |
| num-expert | --num-expert | 专家数量 |
| num-layer | --num-layer | 模型层数 |
| moe-freq | --moe-freq | MoE层频率配置 |
| seq-length | --seq-length | 序列长度 |
| seq-len | --seq-length | 序列长度（别名） |
| dispatcher | --dispatcher | dispatcher 类型 |
| pp-layout | --pipeline-parallel-layout | 流水线布局 |

### 特性映射表

| YAML 特性配置 | train.sh 参数 | 说明 |
|--------------|--------------|------|
| profile: true | --profile | 启用性能分析（nsys） |
| graph: true | --graph | 启用 CUDA Graph |
| offload: [act] | --offload-act | 激活值卸载到 CPU |
| offload: [weight] | --offload-weight | 权重卸载到 CPU |
| offload: [optim] | --offload-optim | 优化器状态卸载到 CPU |
| fine_grained: false | (无参数) | 完全禁用细粒度卸载 |
| fine_grained: [] | --offload-fine --offload-fine-modules [] | 禁用但保持其他参数一致（用于对比测试） |
| fine_grained: [模块列表] | --offload-fine --offload-fine-modules [模块名 ...] | 细粒度卸载指定模块 |

**可用的细粒度卸载模块：** `attn_norm`, `qkv_linear`, `core_attn`, `attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`

**`fine_grained` 的三种模式：**
1. **`false`** - 完全禁用细粒度卸载
2. **`[]`**（空列表）- 禁用细粒度卸载，但保持其他参数与启用时一致（用于 A/B 对比测试）
3. **`[module1, ...]`** - 启用并指定卸载模块（省略时使用默认模块）

使用 `fine_grained` 但不指定模块时（例如在命令行仅用 `--offload-fine`），将使用默认模块：
`attn_norm`, `qkv_linear`, `core_attn`, `attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`

## 矩阵参数展开（Matrix Expansion）

`matrix` 部分定义参数扫描。矩阵中定义的参数会在保持其他参数不变的情况下依次测试：

```yaml
matrix:
  mbs: [1, 2]
  tp: [1, 2]
```

上述配置将生成 4 次测试运行：
1. mbs=1, tp=1
2. mbs=1, tp=2
3. mbs=2, tp=1
4. mbs=2, tp=2

矩阵展开采用**迭代（BFS）方式**实现，而非嵌套循环。这种方式更易于扩展到多个参数维度。

## 日志管理

### 测试前
- 如果存在 `logs/` 目录，将其重命名为 `logs-{时间戳}`
- 创建新的空 `logs/` 目录

### 测试后
- 所有 YAML 测试用例文件会被复制到 `logs/` 目录
- 每次训练产生的日志保存在独立的子目录中

## 添加新的测试用例

1. 在 `testcases/` 目录下创建 YAML 文件
2. 在 `param:` 部分定义基础配置（这些参数在测试中保持不变）
3. 在 `matrix:` 部分定义需要扫描的参数（可选）
4. 在 `feature:` 部分定义需要启用的高级特性（可选）
5. 使用 `./bin/run.sh list` 验证新测试用例是否被识别

### 示例

```yaml
name: my-testcase
description: 我的自定义测试

param:
  pp: 1
  tp: 2
  ep: 4
  global-batch-size: 64
  num-expert: 64
  num-layer: 10
  seq-length: 2048
  dispatcher: hybridep

matrix:
  mbs: [1, 2, 4]

feature:
  profile: false
  graph: true
  offload: [act]
  fine_grained: [attn_norm, moe_act]
```

## 脚本关系图

```
┌─────────────────────────────────────────────────────────┐
│                    用户调用层                            │
├─────────────────────────────────────────────────────────┤
│  ./bin/run.sh list                                       │
│  ./bin/run.sh test <name>                                │
│  ./bin/run.sh -d test <name>   (dry-run模式)              │
└────────────────┬──────────────────────────────────────────┘
                 │
                 ▼ 解析 YAML，展开 matrix，转换 feature
┌─────────────────────────────────────────────────────────┐
│              run.sh - 测试编排层                           │
│  • 扫描 testcases/*.yaml                                 │
│  • 解析 param, matrix, feature 配置                      │
│  • 生成矩阵参数的所有组合（BFS迭代）                      │
│  • 转换 feature 为 train.sh 参数                          │
│  • 管理日志目录                                          │
└────────────────┬──────────────────────────────────────────┘
                 │
                 ▼ 调用 train.sh 执行训练
┌─────────────────────────────────────────────────────────┐
│              train.sh - 训练执行层                         │
│  • 设置环境变量                                          │
│  • 检查并安装 deep_ep                                    │
│  • 构建 torchrun + pretrain_gpt.py 命令                 │
│  • 执行训练                                             │
└─────────────────────────────────────────────────────────┘
```

## 注意事项

1. **参数名展开**：YAML 中使用简写（如 `pp`），run.sh 会自动展开为完整参数名（如 `--pipeline-parallel`）传给 train.sh
2. **Dry-run 模式**：使用 `-d` 或 `--dry-run` 参数可以在不实际执行训练的情况下，查看将要执行的所有命令
3. **矩阵组合数**：多个 matrix 参数的组合数是笛卡尔积，注意组合爆炸（如 4 个参数各有 4 个值 = 256 次运行）
