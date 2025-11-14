# Baseline Experiment Scripts

EMMET 基线复现与评测脚本集合，支持 Memory Replay 机制。

## 📁 脚本说明

| 文件 | 功能 | 用途 | 对应 TODO |
|------|------|------|----------|
| `prepare_data.py` | 数据采样工具 | 从完整数据集中采样指定数量 | Phase 1.1 |
| `run_baseline.py` | **主实验脚本** | 运行单个编辑实验并评测 | 所有 Phase |
| `run_all_baselines.cmd/sh` | **三大基线对比** | ROME vs MEMIT vs EMMET | **Phase 1.2** |
| `run_batch_experiments.py` | 批量实验运行器 | 网格搜索多个配置 | Phase 5.2 |
| `run_lora_ablation.cmd/sh` | **LoRA 消融实验** | 测试不同 rank 的影响 | **Phase 3.2** |
| `run_combined_experiments.cmd` | **组合配置实验** | Replay + LoRA 组合测试 | **Phase 3.2** |
| `analyze_results.py` | 结果分析脚本 | 聚合和统计实验结果 | Phase 5.3 |

## 🚀 快速开始

### 第1步: 三大基线对比（Phase 1.2）

```bash
# Windows
scripts\run_all_baselines.cmd

# Linux
bash scripts/run_all_baselines.sh
```

**目标**: 证明统一框架的必要性与 EMMET 的优势

运行 3 个实验：
- ROME: 单条编辑（batch_size=1），200条
- MEMIT: 批量编辑（batch_size=32），200条
- EMMET: 批量编辑（batch_size=32），200条

**输出**: `results/baseline_comparison/` + `baseline_comparison.csv`

### 第2步: Memory Replay 实验（Phase 2）e 2）

```bash
# 单个 Replay 实验
python scripts\run_baseline.py --method emmet --model gpt2 \
    --num_edits 200 --batch_size 32 --replay_rate 0.3 --seed 42
```

### 第3步: LoRA 消融实验（Phase 3）

```bash
# Windows
scripts\run_lora_ablation.cmd

# Linux
bash scripts/run_lora_ablation.sh
```

测试不同 LoRA rank（4/8/16）对性能的影响。

### 第4步: 组合配置实验

```bash
# Windows
scripts\run_combined_experiments.cmd
```

测试 EMMET + Replay + LoRA 的各种组合配置。

## 📊 实验矩阵概览

### 基线对比实验（TODO 1.2）

**目标**: 证明统一框架必要性

| 实验ID | 方法 | Batch Size | Num Edits | 说明 |
|--------|------|------------|-----------|------|
| 1 | ROME | 1 | 200 | 传统单条编辑 |
| 2 | MEMIT | 32 | 200 | 批量最小二乘 |
| 3 | EMMET | 32 | 200 | 统一闭式解 |

**脚本**: `run_all_baselines.cmd`

### MVP实验矩阵（TODO Phase 2）

**目标**: 验证 Memory Replay 缓解遗忘

根据 TODO.md Phase 2，最小可行实验包括:

| 实验ID | 方法 | Batch Size | Replay Rate | 说明 |
|--------|------|------------|-------------|------|
| 1 | EMMET | 1 | 0.0 | 基线-单条编辑 |
| 2 | EMMET | 32 | 0.0 | 基线-中等批量 |
| 3 | EMMET | 256 | 0.0 | 基线-大批量 |
| 4 | EMMET | 1 | 0.3 | Replay-单条编辑 |
| 5 | EMMET | 32 | 0.3 | Replay-中等批量 |
| 6 | EMMET | 256 | 0.3 | Replay-大批量 |

**固定参数**:
- Model: GPT-2 (774M)
- Num edits: 500
- Seed: 42

## 🔧 脚本详解

### 1. minimal_test.py - 环境验证

```bash
python scripts/minimal_test.py
```

**检查项**:
1. Python 版本 (3.9)
2. PyTorch + CUDA
3. Transformers
4. 其他依赖 (numpy, pandas, scipy)
5. 数据文件
6. GPT-2 模型加载
7. 项目结构
8. 模块导入
9. 数据格式

### 2. run_baseline.py - 主实验脚本

**完整参数**:

```bash
python scripts/run_baseline.py \
    --method emmet \              # 编辑方法: emmet/memit/rome
    --model gpt2 \                # 模型: gpt2/gpt2-xl/llama3.2-3b
    --num_edits 500 \             # 编辑数量
    --batch_size 32 \             # 批量大小
    --replay_rate 0.0 \           # Replay比例 (0-1)
    --use_lora \                  # 启用 LoRA (可选)
    --lora_rank 8 \               # LoRA rank (默认8)
    --lora_alpha 16 \             # LoRA alpha (默认16)
    --seed 42 \                   # 随机种子
    --dataset counterfact_sampled_unique_cf_10_20000 \  # 数据集
    --output_dir results/baseline  # 输出目录
```

**输出结构**:

```
results/baseline/emmet_gpt2_b32_replay0.0_20231113_143052/
├── config.json              # 实验配置
├── experiment.log           # 详细日志
├── edit_results.json        # 编辑过程结果
├── detailed_results.json    # 每条数据的评测结果
├── detailed_results.csv     # CSV格式
├── metrics.json             # 聚合指标 (ES/PS/NS/S)
└── metrics.csv              # CSV格式
```

**评测指标**:

| 指标 | 缩写 | 计算方式 | 含义 |
|------|------|----------|------|
| Efficacy Score | ES | 测试 rewrite prompt | 编辑成功率 |
| Paraphrase Score | PS | 测试 paraphrase prompts | 泛化能力 |
| Neighborhood Specificity | NS | 测试 neighborhood prompts | 知识局部性 |
| Composite Score | S | (ES+PS+NS)/3 | 综合得分 |

### 2. run_all_baselines.cmd - 三大基线对比（TODO 1.2）

```bash
# Windows
scripts\run_all_baselines.cmd

# Linux (创建对应的 .sh 版本)
bash scripts/run_all_baselines.sh
```

**目标**: 证明统一框架的必要性与 EMMET 的优势

**实验配置**:
- Model: GPT-2 XL (1.5B)
- Num edits: 200
- Seed: 42
- ROME: batch_size=1（单条编辑）
- MEMIT: batch_size=32（批量编辑）
- EMMET: batch_size=32（批量编辑）

**对比维度**:
1. **Efficacy Score (ES)**: 编辑成功率
2. **Paraphrase Score (PS)**: 泛化能力
3. **Neighborhood Specificity (NS)**: 知识局部性
4. **时间与显存开销**: 效率对比

**输出结构**:
```
results/baseline_comparison/
├── rome_gpt2-xl_b1_20231114_*/     # ROME 结果
├── memit_gpt2-xl_b32_20231114_*/   # MEMIT 结果
├── emmet_gpt2-xl_b32_20231114_*/   # EMMET 结果
└── baseline_comparison.csv          # 聚合对比表
```

**关键点**（对应 TODO 1.2）:
- ✅ 使用相同数据集与随机种子
- ✅ 对齐评测指标实现
- ✅ 保存中间编辑状态以供后续分析
- ✅ 记录时间与显存开销

### 3. run_mvp_experiments.cmd - MVP实验矩阵（TODO Phase 2）

```bash
# Windows
scripts\run_mvp_experiments.cmd

# Linux
bash scripts/run_mvp_experiments.sh
```

**目标**: 验证 Memory Replay 缓解遗忘

**实验矩阵**: 2种配置 × 3种批量大小 = 6组实验
- EMMET baseline (replay_rate=0.0)
- EMMET + Replay (replay_rate=0.3)
- Batch sizes: 1, 32, 256

**固定参数**:
- Model: GPT-2 (774M)
- Num edits: 500
- Seed: 42

**输出**: `results/baseline/` + 遗忘曲线数据

---

## 🔧 LoRA 集成 (Phase 3)

### LoRA 概述

**Low-Rank Adaptation (LoRA)** 是一种参数高效的微调方法，在 EMMET 编辑后应用。

**核心特性**:
- **后处理式架构**: LoRA 在 EMMET 编辑完成后应用，不修改闭式解
- **低秩分解**: W' = W_base + (α/r) * B @ A
- **参数高效**: 仅增加 r×(d_in + d_out) 个可训练参数（<1%）

### 使用方法

#### 1. 基本用法

```bash
# EMMET + LoRA
python scripts/run_baseline.py \
    --method emmet \
    --model gpt2 \
    --num_edits 100 \
    --batch_size 10 \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16
```

#### 2. 组合使用

```bash
# EMMET + Memory Replay + LoRA
python scripts/run_baseline.py \
    --method emmet \
    --model gpt2 \
    --num_edits 200 \
    --batch_size 16 \
    --replay_rate 0.3 \
    --use_lora \
    --lora_rank 8
```

### LoRA 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_lora` | False | 是否启用 LoRA |
| `--lora_rank` | 8 | 低秩分解的秩（推荐: 4/8/16） |
| `--lora_alpha` | 16.0 | 缩放因子（通常为 2×rank） |

### LoRA 实验脚本

#### run_lora_ablation.cmd/sh - LoRA 消融实验

测试不同 rank 对性能的影响：

```bash
# Windows
scripts\run_lora_ablation.cmd

# Linux
bash scripts/run_lora_ablation.sh
```

**实验配置**:
- EMMET baseline (no LoRA)
- EMMET + LoRA rank=4 (α=8)
- EMMET + LoRA rank=8 (α=16)
- EMMET + LoRA rank=16 (α=32)

**固定参数**: MODEL=gpt2, NUM_EDITS=100, BATCH_SIZE=10, SEED=42

#### run_combined_experiments.cmd - 组合配置实验

测试所有组合配置：

```bash
scripts\run_combined_experiments.cmd
```

**包含 7 种配置**:
1. EMMET baseline
2. EMMET + Replay (0.3)
3. EMMET + LoRA (rank=8)
4. EMMET + Replay (0.3) + LoRA (rank=8)
5. EMMET + Replay (0.5) + LoRA (rank=4)
6. EMMET + Replay (0.3) + LoRA (rank=16)
7. EMMET + Replay (0.1) + LoRA (rank=8)

### LoRA 支持的模型

| 模型 | 默认目标模块 |
|------|--------------|
| GPT-2 | `mlp.c_fc`, `mlp.c_proj` |
| LLaMA | `mlp.up_proj`, `mlp.down_proj`, `mlp.gate_proj` |
| GPT-J | `mlp.fc_in`, `mlp.fc_out` |
| OPT | `fc1`, `fc2` |

### LoRA 参数效率

以 GPT-2 (124M) 为例：

| Rank | LoRA 参数 | 占比 | 训练参数减少 |
|------|-----------|------|--------------|
| 4 | ~0.3M | 0.24% | 99.76% |
| 8 | ~0.6M | 0.48% | 99.52% |
| 16 | ~1.2M | 0.97% | 99.03% |

### LoRA API 参考

```python
from emmet.lora_wrapper import apply_lora_to_edited_model

lora_wrapper = apply_lora_to_edited_model(
    model=edited_model,              # EMMET 编辑后的模型
    target_modules=['mlp.c_fc', 'mlp.c_proj'],  # 目标模块
    rank=8,                          # LoRA rank
    alpha=16.0,                      # 缩放因子
    freeze_base=True                 # 冻结基础参数
)

# 获取参数统计
stats = lora_wrapper.get_param_count()

# 启用/禁用 LoRA
lora_wrapper.enable_lora()
lora_wrapper.disable_lora()

# 合并 LoRA 到基础权重
lora_wrapper.merge_lora()
```

### LoRA 故障排除

**问题：显存不足**
- 减小 LoRA rank (8 → 4)
- 减小 batch_size
- 减少 target_modules 数量

**问题：性能下降**
- 增加 rank (8 → 16)
- 调整 alpha = 2 × rank
- 运行消融实验找到最佳配置

---

### 4. run_batch_experiments.py - 批量实验运行器（TODO 4.2）

**使用配置文件**:

```json
{
  "methods": ["emmet"],
  "models": ["gpt2"],
  "num_edits_list": [500],
  "batch_sizes": [1, 32, 256],
  "replay_rates": [0.0, 0.3],
  "seeds": [42],
  "dataset": "counterfact_sampled_unique_cf_10_20000",
  "output_dir": "results/baseline"
}
```

**运行**:

```bash
# 使用配置文件
python scripts/run_batch_experiments.py --config configs/full_experiment_config.json

# 或使用命令行参数
python scripts/run_batch_experiments.py \
    --methods emmet \
    --models gpt2 \
    --num_edits 500 \
    --batch_sizes 1 32 256 \
    --replay_rates 0.0 0.3 \
    --seeds 42
```

自动运行所有参数组合 (2×3=6 组实验)。

### 5. analyze_results.py - 结果分析（TODO 4.3）

```bash
python scripts/analyze_results.py \
    --results_dir results/baseline \
    --output aggregated_results
```

**输出**:
- `aggregated_results.csv`: 所有实验的详细结果
- `statistics.csv`: 按方法、批量、Replay率分组的统计

**分组维度**:
- Method (emmet/memit/rome)
- Batch Size (1/32/256)
- Replay Rate (0.0/0.3)

### 6. prepare_data.py - 数据采样（TODO 1.1）

```bash
# 采样 200 条
python scripts/prepare_data.py --num 200 --seed 42

# 采样到自定义文件
python scripts/prepare_data.py --num 500 --seed 42 --output data/sample_500.json
```

## 📈 实验工作流（对应 TODO.md）

### Phase 0 (Day 0) - 环境准备与验证

**目标**: 确保技术栈可行性

```bash
# 1. 环境验证
python scripts/minimal_test.py

# 2. 快速测试（10条数据）
scripts\quick_test.cmd
```

**产出**: 环境验证通过 + 快速测试结果

---

### Phase 1 (TODO 1.2) - 三大基线对比

**目标**: 证明统一框架必要性

```bash
# 运行 ROME vs MEMIT vs EMMET 对比
scripts\run_all_baselines.cmd

# 分析结果
python scripts\analyze_results.py --results_dir results/baseline_comparison
```

**产出**: `baseline_comparison.csv` + 对比分析报告

**关键发现**:
- ROME: 精确但慢（单条编辑）
- MEMIT: 快速但近似（最小二乘松弛）
- EMMET: 平衡效率与精度（闭式解）

---

### Phase 2 (TODO Phase 2) - Memory Replay 验证

**目标**: 验证 Replay 机制缓解遗忘

```bash
# 运行 MVP 实验矩阵（6组）
scripts\run_mvp_experiments.cmd

# 分析遗忘曲线
python scripts\analyze_results.py --results_dir results/baseline
```

**产出**: 遗忘曲线图 + Replay 效果分析

---

### Phase 3 (TODO 4.2) - 大规模消融实验

**目标**: 系统评测各配置组合

```bash
# 使用配置文件运行完整实验矩阵
python scripts/run_batch_experiments.py --config configs/full_experiment_config.json
```

**产出**: 完整实验矩阵结果

---

## 📅 实验进度追踪（基于 TODO.md）

### ✅ Phase 0: 知识准备与环境配置

- [x] 环境配置 (conda + PyTorch + Transformers)
- [x] 数据集准备 (CounterFact)
- [x] 环境验证脚本 (`minimal_test.py`)
- [x] 快速测试脚本 (`quick_test.cmd`)

### 🔄 Phase 1: 基线实验与对比 [P0 优先级]

**1.1 小规模快速验证（200-500条）**
- [x] 准备 CounterFact 子集
- [ ] 运行 EMMET 最小示例
- [ ] 确认 ES/PS/NS 指标计算正确
- [ ] 调试超参数

**1.2 三大基线对比实验（ROME / MEMIT / EMMET）**
- [x] 创建 `run_all_baselines.cmd` 脚本
- [ ] ROME: 单条编辑（batch_size=1），200条
- [ ] MEMIT: 批量编辑（batch_size=32），200条
- [ ] EMMET: 批量编辑（batch_size=32），200条
- [ ] 对比三者的 ES/PS/NS 差异
- [ ] 记录时间与显存开销

**产出**: `results/baseline_comparison.csv`

### ⏳ Phase 2: Memory Replay 实现 [P1 核心贡献]

**2.1 Replay Buffer 设计与实现**
- [ ] 设计 Buffer 数据结构
- [ ] 实现采样策略
- [ ] 实现 Buffer 维护

**2.2 集成到 EMMET 闭式解**
- [ ] 在构建约束时拼接当前批 + 历史采样批
- [ ] 数值稳定性处理

**2.3 小规模消融实验**
- [ ] Replay Rate 消融：r ∈ {0, 0.1, 0.3, 0.5}
- [ ] Buffer Size 消融
- [ ] 采样策略对比

### ⏳ Phase 3: 最小化 LoRA 集成 [P2 满足报告承诺]

- [ ] 实现最小 LoRA Wrapper 类
- [ ] 小规模实验：EMMET vs EMMET+LoRA
- [ ] 与 Replay 组合验证

### ⏳ Phase 4: 中大规模系统实验 [P3 证明有效性]

**4.1 扩展到中规模数据集（2000-5000条）**
- [ ] 观察渐进遗忘 → 灾难遗忘的转折点
- [ ] 多种配置对比

**4.2 批量规模消融实验**
- [ ] 批量大小：{1, 8, 32, 128, 512, 1024}
- [ ] Replay 比例：r ∈ {0, 0.1, 0.3, 0.5}
- [ ] 随机种子：{1, 2, 3}

**4.3 可视化与分析**
- [ ] 遗忘曲线图
- [ ] 批量规模对比图
- [ ] Replay 效果热力图

### ⏳ Phase 5: 报告撰写与文档整理 [P4 最终交付]

- [ ] 技术报告撰写（ACL 格式）
- [ ] 代码文档与可复现性
- [ ] 实验日志与结果归档

---

## 🗓️ 已完成实验记录

### Day 0 (11/13) - 环境准备 [✅ 完成]

```bash
# 1. 验证环境
python scripts/minimal_test.py

# 2. 快速测试
python scripts/test_baseline.py
# 或使用便携脚本
scripts\quick_test.cmd  # Windows
bash scripts/quick_test.sh  # Linux

# 3. 准备数据（可选）
python scripts/prepare_data.py --num 500 --seed 42
```

### Day 1-2 (11/14-15) - EMMET基线

```bash
# 选项1: 运行完整MVP矩阵
scripts\run_mvp_experiments.cmd  # Windows
bash scripts/run_mvp_experiments.sh  # Linux

# 选项2: 手动运行单个实验
python scripts/run_baseline.py --method emmet --model gpt2 --num_edits 500 --batch_size 32 --seed 42

# 选项3: 批量配置
python scripts/run_batch_experiments.py --config configs/full_experiment_config.json
```

### Day 3-4 (11/16-17) - Memory Replay

**需要实现**:
1. 创建 `src/emmet/replay_buffer.py`
2. 修改 `src/emmet/emmet_main.py` 集成 Replay
3. 运行对比实验:

```bash
# Replay实验（replay_rate=0.3已在MVP矩阵中）
python scripts/run_baseline.py --method emmet --model gpt2 --num_edits 500 --batch_size 32 --replay_rate 0.3 --seed 42
```

### Day 5 (11/18) - 结果分析与报告

```bash
# 1. 聚合结果
python scripts/analyze_results.py --results_dir results/baseline

# 2. 查看统计
# 打开 results/baseline/statistics.csv

# 3. 生成图表（需额外脚本）
# 绘制 ES/PS/NS 对比
# 绘制 Batch Size 影响
# 绘制 Replay Rate 影响
```

## ⚙️ 配置说明

### 环境要求

- Python: 3.9.7
- PyTorch: 1.12.1 (CUDA 11.3)
- Transformers: 4.23.1
- CUDA: 11.3 (可选，推荐)
- GPU 显存: 2GB+ (GPT-2), 6GB+ (GPT-2-XL)

### 性能预估

| 配置 | 时间 | 显存 |
|------|------|------|
| 500条, batch=1, GPT-2 | ~5小时 | 2GB |
| 500条, batch=32, GPT-2 | ~1小时 | 4GB |
| 500条, batch=256, GPT-2 | ~30分钟 | 8GB |

**加速建议**:
- 使用更大的 batch_size (如果显存允许)
- 使用 CUDA (比 CPU 快 10-50倍)
- 减少 num_edits 用于快速测试

## 🐛 故障排查

### 问题1: CUDA out of memory

```bash
# 解决方案1: 减小批量
python scripts/run_baseline.py ... --batch_size 1

# 解决方案2: 使用CPU（Windows）
set CUDA_VISIBLE_DEVICES=-1
python scripts/run_baseline.py ...

# 解决方案2: 使用CPU（Linux）
CUDA_VISIBLE_DEVICES=-1 python scripts/run_baseline.py ...
```

### 问题2: ModuleNotFoundError

```bash
# 确保在项目根目录
cd d:\Projects\nlp_final_project\emmet-stability-replay  # Windows
cd /path/to/emmet-stability-replay  # Linux

# 检查conda环境
conda env list
conda activate emmet-edit
```

### 问题3: 数据文件未找到

```bash
# 检查数据文件
dir data\counterfact_sampled_unique_cf_10_20000.json  # Windows
ls data/counterfact_sampled_unique_cf_10_20000.json  # Linux

# 如果缺失，使用prepare_data.py生成样本
python scripts/prepare_data.py --num 500 --seed 42
```

### 问题4: 模型下载失败

```bash
# 手动下载模型
python scripts/download_models.py

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com  # Linux
set HF_ENDPOINT=https://hf-mirror.com  # Windows
```

### 问题5: Replay功能不可用

**预期行为**: Day 3-4 之前，`--replay_rate` 参数会被记录但不生效。

**解决**: 按 todo.md 计划，在 Day 3-4 实现 `src/emmet/replay_buffer.py`。

## 📝 实现清单

### ✅ 已完成 (Day 0)

- [x] `minimal_test.py` - 环境验证脚本
- [x] `run_baseline.py` - 主实验脚本 (完整评测逻辑)
- [x] `prepare_data.py` - 数据采样工具
- [x] `test_baseline.py` - 快速测试脚本
- [x] `run_batch_experiments.py` - 批量实验运行器
- [x] `analyze_results.py` - 结果分析脚本
- [x] `quick_test.cmd/sh` - 便携测试脚本
- [x] `run_mvp_experiments.cmd/sh` - MVP实验矩阵脚本
- [x] 文档合并 (README.md)

### ⏰ 待实现 (Day 1-5)

- [ ] Day 1-2: 运行EMMET基线实验 (500条)
- [ ] Day 3-4: 实现 Memory Replay
  - [ ] `src/emmet/replay_buffer.py`
  - [ ] 修改 `src/emmet/emmet_main.py`
- [ ] Day 3-4: 运行 Replay 对比实验
- [ ] Day 5: 结果可视化脚本
- [ ] Day 5: 撰写实验报告

## 🔗 相关文档

- `todo.md` - 项目时间线和任务清单
- `docs/experiment_scripts.md` - 实验脚本详细文档
- `docs/init_guide.md` - 初始化指南
- `configs/full_experiment_config.json` - 实验配置示例

## 🎯 核心特性

### 1. 完整的评测指标

- ✅ ES (Efficacy Score) - 编辑成功率
- ✅ PS (Paraphrase Score) - 泛化能力
- ✅ NS (Neighborhood Specificity) - 知识局部性
- ✅ S (Composite Score) - 综合得分

### 2. 灵活的批量处理

- 支持 batch_size = 1, 32, 256 等
- 自动处理不能整除的最后一批
- 显存自适应（batch大小影响显存）

### 3. 可复现性保证

- 固定随机种子 (--seed 42)
- 完整的实验配置保存
- 详细的运行日志

### 4. 多格式输出

- JSON: 详细的嵌套结构
- CSV: 方便 Excel/Pandas 分析
- 日志: 实时调试信息

### 5. Memory Replay 接口

- `--replay_rate` 参数预留
- Day 3-4 实现后即可使用
- 无需修改主实验脚本

## 🤝 贡献

这是 NLP 课程期末项目的实验脚本集合。

**项目目标**: 
- 复现 EMMET 基线
- 实现 Memory Replay 机制
- 对比批量大小和 Replay 率的影响

**截止日期**: 2025-11-18

---

**最后更新**: 2025-11-13
**状态**: Day 0 完成，所有基础脚本就绪 ✅
