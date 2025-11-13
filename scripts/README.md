# Baseline Experiment Scripts

EMMET 基线复现与评测脚本集合，支持 Memory Replay 机制。

## 📁 脚本说明

| 文件 | 功能 | 用途 |
|------|------|------|
| `minimal_test.py` | 环境验证（9项检查） | Day 0: 确保环境配置正确 |
| `prepare_data.py` | 数据采样工具 | 从完整数据集中采样指定数量 |
| `test_baseline.py` | 快速测试（10条数据） | 验证脚本是否正常工作 |
| `run_baseline.py` | 主实验脚本 | 运行单个编辑实验并评测 |
| `run_batch_experiments.py` | 批量实验运行器 | 网格搜索多个配置 |
| `analyze_results.py` | 结果分析脚本 | 聚合和统计实验结果 |
| `quick_test.cmd/sh` | 快速测试便携脚本 | Windows/Linux快速验证 |
| `run_mvp_experiments.cmd/sh` | MVP实验矩阵 | 运行完整的6组基线实验 |

## 🚀 快速开始

### 第0步: 环境验证

```bash
# Windows
cd d:\Projects\nlp_final_project\emmet-stability-replay
conda activate emmet-edit
python scripts\minimal_test.py

# Linux
cd /path/to/emmet-stability-replay
conda activate emmet-edit
python scripts/minimal_test.py
```

验证项目:

- ✅ Python 3.9
- ✅ PyTorch 1.12.1 + CUDA
- ✅ Transformers 4.23.1
- ✅ 数据文件存在
- ✅ GPT-2 模型可加载
- ✅ 项目模块可导入

### 第1步: 快速测试

```bash
# Windows
scripts\quick_test.cmd

# Linux
bash scripts/quick_test.sh
```

运行 10 条数据的小规模测试，验证完整流程。

### 第2步: 运行MVP实验

```bash
# Windows
scripts\run_mvp_experiments.cmd

# Linux
bash scripts/run_mvp_experiments.sh
```

自动运行 6 组实验（见下文实验矩阵）。

## 📊 实验矩阵 (MVP)

根据 todo.md 第7节，最小可行实验包括:

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

### 3. run_batch_experiments.py - 批量实验

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

### 4. analyze_results.py - 结果分析

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

### 5. prepare_data.py - 数据采样

```bash
# 采样 200 条
python scripts/prepare_data.py --num 200 --seed 42

# 采样到自定义文件
python scripts/prepare_data.py --num 500 --seed 42 --output data/sample_500.json
```

## 📈 实验工作流

### Day 0 (今天 11/13) - 环境准备

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
