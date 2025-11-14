# 项目任务清单（基于 EMMET 统一编辑框架与稳定性增强）

> **核心目标**: 在 `unified-model-editing` 基础上，完成 EMMET 基线复现 + Memory Replay 稳定性增强 + 最小化 LoRA 集成 + 系统评测与报告撰写。

## 项目验收标准

**必须完成的产物**:

- ✅ ROME/MEMIT 二大基线的对比实验结果（证明统一框架的必要性）
- ✅ EMMET 基线复现（作为改进的框架基础）
- ✅ EMMET + Memory Replay 实现与遗忘缓解实验（核心贡献）
- ✅ 最小化 LoRA 集成的概念验证（满足报告承诺）
- ✅ 中规模数据集实验（2000-5000条，观察渐进/灾难遗忘现象）
- ✅ 可复现的实验脚本 + 结果可视化 + ACL 格式技术报告

**关键指标与可视化**:

- ES (Efficacy Score) / PS (Paraphrase Score) / NS (Neighborhood Score)
- 遗忘曲线（编辑序列步数 vs NS/历史编辑保留率）
- 批量规模对比（1/8/32/128/512/1024）
- 顺序编辑 vs 批量编辑 vs Replay 增强的三方对比

---

## 任务优先级体系

### P0 - 核心基线（必须完成）

**目标**: 建立实验基础，验证技术栈可行性

### P1 - 稳定性增强（核心贡献）

**目标**: 实现并验证 Memory Replay 缓解遗忘

### P2 - PEFT 集成（满足报告承诺）

**目标**: 最小化 LoRA 实现，作为第二种稳定性方案

### P3 - 系统评测（证明有效性）

**目标**: 中大规模实验，对比分析，可视化

### P4 - 报告撰写（最终交付）

**目标**: ACL 格式技术报告与代码文档

---

## 📚 Phase 0: 知识准备与环境配置

### 0.1 理论基础学习

**理论/算法**:

- [ ] ROME 等式约束编辑的基本推导与目标函数
- [ ] MEMIT 的最小二乘放宽、批量编辑思想
- [ ] EMMET 的统一视角与闭式解（KKT 条件、约束最小二乘）
- [ ] 遗忘现象：渐进遗忘 vs 灾难遗忘的区别与成因

**工程/实践**:

- [ ] `unified-model-editing` 代码结构梳理：
  - 数据集管线（CounterFact/ZSRE）
  - `emmet/memit/rome` 各模块功能
  - `util/nethook.py` 钩子机制
  - 评测脚本与指标实现
- [ ] PyTorch Hook/Forward Intervene 技巧
- [ ] Hugging Face Transformers 层命名与权重注入位置
- [ ] PEFT（LoRA）基本原理与库用法
- [ ] 实验可复现性：随机种子、批量大小、显存管理

### 0.2 环境配置

- [x] 使用 conda 创建环境并安装依赖（优先 `conda`，必要时再补 `pip`）
- [x] 校验 Python 版本与 PyTorch/CUDA
- [x] 验证模型加载与最小推理脚本
- [x] 准备数据集：验证 `data/counterfact_*.json` 与采样脚本
- [x] 准备基础模型（GPT-2 XL / Llama-2-7B）

**产出**: `env.yaml` + 模型加载验证日志

---

## 🔬 Phase 1: 基线实验与对比 [P0 优先级]

### 1.1 小规模快速验证（200-500条）

**任务**:

- [x] 准备 CounterFact 子集（200条用于调试）
- [x] 运行 EMMET 最小示例，验证代码可用性
- [x] 确认 ES/PS/NS/GE/S 指标计算正确
- [x] 调试超参数（学习率、层选择、批量大小）

**产出**: `results/quick_validation.csv` + 调试日志

### 1.2 三大基线对比实验（ROME / MEMIT / EMMET）

**目标**: 证明统一框架的必要性与 EMMET 的优势

**任务**:

- [x] ROME: 单条编辑（batch_size=1），200条
- [x] MEMIT: 批量编辑（batch_size=32），200条
- [x] EMMET: 批量编辑（batch_size=32），200条
- [x] 对比三者的 ES/PS/NS 差异
- [x] 记录时间与显存开销

**产出**: `scripts/run_all_baselines.cmd` + `scripts/run_all_baselines.sh` + `results/baseline_comparison.csv`

**关键点**:

- 使用相同数据集与随机种子
- 对齐评测指标实现
- 保存中间编辑状态以供后续分析

---

## 🔄 Phase 2: Memory Replay 实现 [P1 核心贡献]

### 2.1 Replay Buffer 设计与实现

**任务**:

- [x] 设计 Buffer 数据结构
  - 存储：(subject, relation, object) + 原始/改写提示
  - 可选：缓存中间统计（Keys/Values）以加速
- [x] 实现采样策略
  - 随机采样 vs 优先采样（按编辑时间/重要性）
  - 采样比例参数化（replay_rate: 0.1/0.3/0.5）
- [x] 实现 Buffer 维护
  - 插入新编辑（成功后更新）
  - 容量限制与淘汰策略（FIFO/LRU）
  - 去重处理

**产出**: `src/emmet/replay_buffer.py` ✅

### 2.2 集成到 EMMET 闭式解

**任务**:

- [x] 阅读 `emmet/compute_ks.py` 与 `compute_z.py`
- [x] 定位等式约束构建步骤
- [x] 在构建约束时拼接当前批 + 历史采样批
  - 确保维度一致
  - 权重调整（可选：历史样本降权）
- [x] 数值稳定性处理
  - Tikhonov 正则化
  - 条件数监控
  - 异常检测与回退机制

**产出**: `src/emmet/emmet_replay.py` + `src/emmet/replay_utils.py` + 修改 `scripts/run_baseline.py` 支持 `--replay_rate` 参数 ✅

### 2.3 小规模消融实验

**任务**:

- [ ] Replay Rate 消融：r ∈ {0, 0.1, 0.3, 0.5}
- [ ] Buffer Size 消融：max_items ∈ {50, 100, 200}
- [ ] 采样策略对比：Random vs Priority
- [ ] 观察 NS 与历史编辑保留率的变化

**数据**: 500条顺序编辑（观察遗忘曲线）

**产出**: `results/replay_ablation.csv` + 遗忘曲线图

---

## 🔧 Phase 3: 最小化 LoRA 集成 [P2 满足报告承诺]

### 3.1 后处理式 LoRA 实现（简化方案）

**设计思路**: 不修改 EMMET 闭式解，仅在编辑后的权重上添加 LoRA 层作为"微调适配器"

**任务**:

- [ ] 实现最小 LoRA Wrapper 类

  ```python
  class MinimalLoRAWrapper:
      def __init__(self, edited_weight, rank=8):
          # 在 EMMET 编辑后的权重基础上添加低秩调整
          self.base_weight = edited_weight.detach()
          self.lora_A = torch.randn(...) * 0.01
          self.lora_B = torch.randn(...) * 0.01
  ```

- [ ] 为目标层（MLP/Attention）添加 LoRA 开关
- [ ] 验证参数量与显存占用下降
- [ ] 小规模实验（50-100条）：EMMET vs EMMET+LoRA

**产出**: `src/emmet/lora_wrapper.py` + `--use_lora` 参数

**时间预估**: 6-8 小时（实现 + 调试 + 验证）

### 3.2 与 Replay 组合验证

**任务**:

- [ ] EMMET + Replay + LoRA 三种配置对比
- [ ] 观察 NS 稳定性与参数效率的权衡
- [ ] 记录 LoRA rank 对性能的影响（rank={4,8,16}）

**产出**: `results/lora_ablation.csv`

---
## 🔧 Phase 4 编辑信用评分与保险机制 (Edit Trust / Rollback)

**目标**: 为顺序/批量编辑过程加入实时风险控制，降低副作用与误编辑累积；在置信不足时自动回滚或缩放编辑。

**核心概念**:
- Trust Score = 加权(编辑成功增益) − 惩罚(副作用/困惑度上升/邻域串扰/过度置信漂移)
- Insurance = 回滚 (硬) 或 Delta 缩放 (软)，并记录触发原因
- 评价 = 减少 NS 恶化率 + 提高历史编辑保留率 + 控制误报/漏报

**指标定义**:
- edit_success_gain: 目标事实目标 token 平均对数概率提升
- heldout_ppl_delta: Held-out 文本平均 PPL 上升值
- neighbor_flip_rate: 相似实体断言被错误提升的比例
- confidence_shift: max softmax 过度提升幅度
- (可选) attribution_focus_delta: 编辑层梯度/IG 聚焦度变化

**任务**:
- [ ] 设计 TrustMetrics 数据结构与组合函数 (0-1 归一化)
- [ ] 实现 `src/emmet/trust.py`（预评估 + 后评估）
- [ ] 在 `EMMETHyperParams` 增加字段：`trust_enable`, `trust_threshold`, `trust_weights`, `trust_heldout_samples`
- [ ] 修改 `emmet_main.execute_emmet`：注入前备份 → 编辑 → 评估 → 低分回滚或缩放
- [ ] 增加 CLI 参数：`--trust-enable --trust-threshold --trust-heldout-samples`
- [ ] 采样 Held-out 集：从 CounterFact 非目标条目抽取 N 条
- [ ] 记录 JSON 行：每次编辑的 metrics + decision + rollback_flag
- [ ] 统计：rollback_rate / false_positive_rate / false_negative_rate
- [ ] 阈值消融：threshold ∈ {0.2,0.3,0.4,0.5,0.6}
- [ ] 可视化：ROC/PR 曲线 + rollback 对历史编辑保留率影响
- [ ] 出错保护：评估异常时跳过回滚并打日志

**产出**:
- 代码: `src/emmet/trust.py` + `emmet_main.py` 补丁
- 配置: 更新 hparams JSON 加入 trust 字段
- 结果: `results/trust_eval.csv`, `results/trust_events.jsonl`
- 图表: `figs/trust_roc.png`, `figs/rollback_effect_curve.png`
- 文档: `docs/trust_mechanism.md`

**成功判据 (建议)**:
- 历史编辑保留率相对无保险提升 ≥ 8%
- rollback 决策准确率 (1 - 误报/漏报平均) ≥ 0.75
- 额外时间开销 ≤ 20%（gpt2-xl, 500 顺序编辑）
- 副作用指标 (average heldout_ppl_delta) 降低 ≥ 30%

**扩展 (后续可选)**:
- Soft-scaling：分段线性缩放 delta 而非全回滚
- 动态阈值：基于最近窗口均值自适应调整
- 标注集微调：用少量人工标注的 “应回滚” 样本学习分类器

---

## 📊 Phase 5: 中大规模系统实验 [P3 证明有效性]

### 5.1 扩展到中规模数据集（2000-5000条）

**目标**: 观察渐进遗忘 → 灾难遗忘的转折点

**任务**:

- [ ] 准备 CounterFact 2000-5000 条子集
- [ ] 顺序编辑实验（观察遗忘曲线）
- [ ] EMMET baseline vs EMMET+Replay vs EMMET+LoRA vs EMMET+Replay+LoRA
- [ ] 记录每 100-200 条编辑后的指标快照

**产出**: `results/large_scale_forgetting.csv` + 遗忘曲线图

### 5.2 批量规模消融实验

**实验矩阵**:

- [ ] 批量大小：{1, 8, 32, 128, 512, 1024}
- [ ] Replay 比例：r ∈ {0, 0.1, 0.3, 0.5}
- [ ] LoRA rank：{4, 8, 16}（仅 EMMET+LoRA 配置）
- [ ] 随机种子：{1, 2, 3}（确保可复现性）

**任务**:

- [ ] 编写批处理脚本 `scripts/run_batch_experiments.py`
- [ ] 自动化运行所有配置组合
- [ ] 聚合结果到 `results/ablation_matrix.csv`

**产出**: 批处理脚本 + 完整实验矩阵结果

### 5.3 可视化与分析

**任务**:

- [x] 遗忘曲线图（编辑步数 vs NS/历史保留率）
- [x] 批量规模对比图（batch_size vs ES/PS/NS）
- [ ] Replay 效果热力图（replay_rate × batch_size）
- [ ] 方法对比雷达图（多指标综合展示）

**产出**: `src/utils/visualize.py` + `scripts/analyze_results.py` (已集成可视化) + `figs/*.png` ✅

---

## 📝 Phase 6: 报告撰写与文档整理 [P4 最终交付]

### 6.1 技术报告撰写（ACL 格式）

**章节结构**:

- [ ] **Introduction**: 研究背景与动机（model editing + forgetting 问题）
- [ ] **Related Work**: ROME/MEMIT/EMMET + PEFT + Memory Replay 文献综述
- [ ] **Method**:
  - EMMET 统一框架回顾
  - Memory Replay 机制设计与实现
  - 最小化 LoRA 集成方案
- [ ] **Experiments**:
  - 实验设置（数据集、模型、超参数）
  - 基线对比实验
  - 消融实验（Replay rate / LoRA rank / batch size）
  - 遗忘曲线分析
- [ ] **Results & Discussion**:
  - 定量结果表格
  - 可视化图表分析
  - 失败案例与局限性讨论
- [ ] **Conclusion & Future Work**

**产出**: `report/final_report.pdf` (ACL 格式)

### 6.2 代码文档与可复现性

**任务**:

- [x] 更新 `README.md`：项目介绍、环境配置、快速开始
- [x] 补充 `docs/experiment_scripts.md`：所有实验的运行命令
- [x] 代码注释完善：关键函数与算法步骤
- [ ] 创建 `environment.yml` / `requirements.txt`（固定版本）
- [x] 添加 `scripts/reproduce_all.cmd`（一键复现所有实验）- 已有 `run_all_baselines.cmd/sh`

**产出**: 完整代码文档 + 可复现脚本

### 6.3 实验日志与结果归档

**任务**:

- [ ] 整理所有实验日志到 `logs/`
- [ ] 汇总所有结果 CSV 到 `results/`
- [ ] 保存最佳模型检查点（如有）
- [ ] 创建 `COMPLETION_SUMMARY.md`：任务完成情况总结

**产出**: 完整实验归档 + 完成总结

---

## 🚨 风险管理与备选方案

### 显存/计算资源不足

- **备选方案 1**: 使用 GPT-2 (medium/large) 替代 GPT-2 XL
- **备选方案 2**: 降低批量大小，分批处理
- **备选方案 3**: 启用 8-bit 量化（`bitsandbytes`）
- **备选方案 4**: 分层编辑（逐层处理，避免全模型同时加载）

### 数值不稳定问题

- **解决方案 1**: Tikhonov 正则化（增加阻尼项）
- **解决方案 2**: 限制 Replay Buffer 大小，避免矩阵过大
- **解决方案 3**: 监控条件数，异常时回退到无 Replay 模式
- **解决方案 4**: 使用稳定的线性求解器（`torch.linalg.lstsq`）

### 实验时间不足

- **优先级调整**: 确保 P0/P1 完成，P2 简化为概念验证
- **并行策略**: 多个实验配置同时运行（多 GPU/多进程）
- **结果复用**: 从文献引用部分基准数据，减少重复实验

### 复现性问题

- **强制措施**:
  - 所有实验固定随机种子（`torch.manual_seed(42)`）
  - 记录完整环境信息（`conda list --export`）
  - 保存超参数配置文件（JSON）
  - 保留失败日志用于分析

---

## 附录：快速参考命令（Windows CMD）

```cmd
:: 创建并激活环境（优先 conda）
conda create -n emmet-edit python=3.10 -y
conda activate emmet-edit

:: 安装依赖（示例，具体以仓库为准）
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
pip install -r unified-model-editing/requirements.txt

:: 运行快速测试（10条编辑）
python scripts\run_baseline.py --method emmet --model gpt2-xl --num_edits 10 --seed 42

:: 运行完整基线对比
scripts\run_all_baselines.cmd

:: 数据采样（示例）
python unified-model-editing\create_samples_cf.py --n_samples 200

:: 评测与分析
python scripts\analyze_results.py --results_dir results\
```
