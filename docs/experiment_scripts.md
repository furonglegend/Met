# 实验脚本使用指南

本文档说明如何使用实验脚本进行 ROME/MEMIT/EMMET 基线实验。

---

## 脚本列表

### Python 脚本

1. **run_baseline.py** - 运行单个实验
2. **run_batch_experiments.py** - 批量运行多个实验
3. **analyze_results.py** - 分析和聚合实验结果
4. **download_models.py** - 下载模型权重

### 便捷脚本

5. **quick_test.cmd / quick_test.sh** - 快速测试（10条数据）
6. **run_all_baselines.cmd / run_all_baselines.sh** - 运行所有基线实验

---

## 1. run_baseline.py - 单实验运行器

### 基本用法

```bash
# Windows
python scripts\run_baseline.py --method emmet --model gpt2-xl --num_edits 100 --seed 42

# Linux
python scripts/run_baseline.py --method emmet --model gpt2-xl --num_edits 100 --seed 42
```

### 参数说明

- `--method` - 编辑方法：rome, memit, emmet
- `--model` - 模型名称：gpt2, gpt2-xl, llama3.2-3b
- `--num_edits` - 编辑数量（默认：100）
- `--seed` - 随机种子（默认：42）
- `--batch_size` - 批量大小（默认：1）
- `--dataset` - 数据集名称（默认：counterfact_sampled_unique_cf_10_20000）
- `--output_dir` - 输出目录（默认：results/baseline）

### 输出文件

实验结果保存在 `results/baseline/方法_模型_时间戳/` 目录下：

- `config.json` - 实验配置
- `results.json` - 详细结果
- `metrics.json` - 评测指标
- `metrics.csv` - CSV 格式指标
- `experiment.log` - 运行日志

---

## 2. run_batch_experiments.py - 批量实验运行器

### 使用配置文件

```bash
python scripts/run_batch_experiments.py --config configs/batch_config_example.json
```

### 使用命令行参数

```bash
python scripts/run_batch_experiments.py \
    --methods rome memit emmet \
    --models gpt2-xl \
    --num_edits 100 500 \
    --seeds 42 43 44
```

### 配置文件格式

```json
{
  "methods": ["rome", "memit", "emmet"],
  "models": ["gpt2-xl"],
  "num_edits": [100, 500],
  "seeds": [42, 43, 44]
}
```

---

## 3. analyze_results.py - 结果分析工具

### 基本用法

```bash
# 分析指定目录的所有结果
python scripts/analyze_results.py --results_dir results/baseline

# 指定输出文件
python scripts/analyze_results.py --results_dir results --output analysis.csv
```

### 输出文件

- `analysis_时间戳.csv` - 聚合结果
- `analysis_时间戳_stats.csv` - 统计数据
- `analysis_时间戳.txt` - 文本报告

---

## 4. download_models.py - 模型下载工具

### 基本用法

```bash
# 下载 GPT-2-XL
python scripts/download_models.py --model gpt2-xl

# 下载 Llama-3.2-3B
python scripts/download_models.py --model llama3.2-3b

# 指定缓存目录
python scripts/download_models.py --model gpt2-xl --cache-dir ./models
```

### 可用模型

- `gpt2` (124M)
- `gpt2-xl` (1.5B)
- `llama3.2-1b` (1B)
- `llama3.2-3b` (3B)
- `llama2-7b` (7B，需要认证)

---

## 5. quick_test - 快速测试脚本

运行一个最小实验（10条数据）来测试环境配置。

### Windows

```cmd
scripts\quick_test.cmd
```

### Linux

```bash
chmod +x scripts/quick_test.sh
./scripts/quick_test.sh
```

---

## 6. run_all_baselines - 运行所有基线

运行 ROME/MEMIT/EMMET 三种方法并分析结果。

### Windows

```cmd
scripts\run_all_baselines.cmd
```

### Linux

```bash
chmod +x scripts/run_all_baselines.sh
./scripts/run_all_baselines.sh
```

### 执行步骤

1. 激活 conda 环境
2. 运行批量实验（3种方法）
3. 自动分析结果
4. 生成汇总报告

---

## 使用示例

### 示例 1：快速验证

```bash
# 1. 运行快速测试
./scripts/quick_test.sh

# 2. 查看结果
ls results/baseline/
```

### 示例 2：单方法评测

```bash
# 运行不同编辑数量的实验
python scripts/run_baseline.py --method emmet --model gpt2-xl --num_edits 100 --seed 42
python scripts/run_baseline.py --method emmet --model gpt2-xl --num_edits 500 --seed 42

# 分析结果
python scripts/analyze_results.py --results_dir results/baseline
```

### 示例 3：完整基线对比

```bash
# 使用配置文件运行批量实验
python scripts/run_batch_experiments.py --config configs/batch_config_example.json

# 分析结果
python scripts/analyze_results.py --results_dir results/baseline
```

### 示例 4：论文实验

```bash
# 运行完整实验矩阵
python scripts/run_batch_experiments.py --config configs/full_experiment_config.json

# 这将运行：3 种方法 × 2 个模型 × 6 种编辑数 × 3 个种子 = 108 个实验
```

---

## 常见问题

### 1. 找不到模块

确保激活了正确的 conda 环境：

```bash
conda activate emmet-edit
```

### 2. 显存不足

减少批量大小或编辑数量：

```bash
python scripts/run_baseline.py --method emmet --model gpt2-xl --num_edits 10 --batch_size 1
```

### 3. 缺少配置文件

检查超参数配置是否存在：

```bash
# Windows
dir src\hparams\EMMET\
dir src\hparams\ROME\
dir src\hparams\MEMIT\

# Linux
ls src/hparams/EMMET/
ls src/hparams/ROME/
ls src/hparams/MEMIT/
```

---

**最后更新**: 2025年11月12日
