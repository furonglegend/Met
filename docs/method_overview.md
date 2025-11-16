# 模型编辑方法与本项目中的角色概览

> 本文档简要说明本项目中使用的几个关键方法/框架：ROME、MEMIT、EMMET、LoRA、Memory Replay 和消融实验，以及它们在本仓库中的具体作用。

---

## 1. ROME（Rank-One Model Editing）

### 1.1 基本原理（直觉）

- **目标**：在不大幅破坏模型其它行为的前提下，修改某一条具体事实（如“巴黎是法国首都”）。
- **核心思路**：
  - 找到模型内部最“负责”这条事实的某一层 MLP（通常是中间层 FFN）。
  - 在该层附近做一阶线性近似：`f(h) ≈ W h + b`，其中 `h` 是某个特定输入表示。
  - 希望把输出从旧目标 `y_old` 改为新目标 `y_new`，同时对其它输入的影响最小。
  - 推导得到一个 **秩 1（rank-one）更新**：`ΔW = u v^T`，解一个小规模线性方程，使得：
    - 针对目标样本：输出朝 `y_new` 方向移动；
    - 针对其它样本：输出变化尽量小（近似最小二乘意义下的“局部修改”）。

### 1.2 在本项目中的角色

- **定位**：经典的单事实编辑基线方法，用来与更复杂方法比较。
- **代码位置**：
  - 实现：`src/rome/` 目录。
  - 超参配置：`src/hparams/ROME/` 下按模型划分的 JSON。
- **如何被调用**：
  - 通过脚本 `scripts/run_baseline.py`，设置 `--method rome` 即可跑 ROME 方案。
  - 在 `scripts/run_all_baselines.*` 中，与 MEMIT / EMMET / EMMET+Replay 等一起跑批量对比实验。
- **作用**：提供一个“局部 rank-one 更新”的简单基线，帮助评估 MEMIT / EMMET / Replay 等设计带来的增益。

---

## 2. MEMIT（Mass-Editing Memory in a Transformer）

### 2.1 基本原理

- **目标**：一次性对多条事实进行联合编辑，同时保持模型整体行为尽量稳定。
- **相对 ROME 的扩展**：
  - ROME 更适合单条编辑，多次叠加可能互相干扰；
  - MEMIT 将多条编辑视为一个联合优化问题，通过线性化和统计量一次性求出联合更新。
- **核心思路（简化版）**：
  - 仍然在若干选定层的 MLP 权重上做线性近似；
  - 为每条编辑构造约束：给定输入表示，希望输出偏移到目标方向；
  - 使用预先估计的 **二阶统计（mom2）** 控制更新的规模与方向；
  - 解一个多目标的最小二乘问题，得到一次性的 `ΔW`，实现多事实联合编辑。

### 2.2 在本项目中的角色

- **定位**：面向多事实编辑的强基线方法。
- **代码位置**：
  - 实现：`src/memit/`。
  - 超参配置：`src/hparams/MEMIT/`。
- **如何被调用**：
  - `scripts/run_baseline.py` 中通过 `--method memit` 跑 MEMIT 流程。
  - 结果与 ROME / EMMET 一起汇总到 `results/` 下的各类对比表和可视化图中。
- **作用**：作为“多事实编辑 + 稳定性控制”的已知强基线，为 EMMET 及本项目扩展（Replay、LoRA 等）的性能对比提供参考点。

---

## 3. EMMET（本项目的核心编辑方法）

### 3.1 基本原理（结合代码直觉）

> 本仓库中的 EMMET 实现在 MEMIT 思路之上，加入更灵活的优化目标与稳定性控制，是后续一切扩展（Replay、LoRA-native、Trust/Rollback）的基础。

- **目标**：
  - 提高编辑的**成功率（efficacy）**；
  - 更好约束更新范数/数值稳定性；
  - 允许与 LoRA、Trust/Rollback 等机制平滑集成。
- **主要要素（见 `src/emmet/emmet_hparams.py`）**：
  - 层与模块控制：`layers`, `layer_selection`, 各种 `*_module_tmp` 模板；
  - 优化超参：`v_num_grad_steps`, `v_lr`, `v_loss_layer`, `v_weight_decay` 等；
  - 正则与目标：`clamp_norm_factor`, `kl_factor`, `update_norm_lambda`, `emmet_lambda`；
  - 扩展功能开关：
    - LoRA-native 模式：`edit_mode`（`"raw"` / `"lora_native"`）、`lora_rank`, `lora_alpha`, `lora_scale`, `lora_use_svd`, `lora_fit_steps`, `allow_fallback` 等；
    - Trust/Rollback：`trust_enable`, `trust_threshold`, `trust_action`, `trust_scale`, `trust_heldout_samples`, `trust_weights`。
- **实现层面（见 `src/emmet/emmet_main.py`）**：
  - 从选定层抽取待编辑权重，保存备份；
  - 通过 `execute_emmet` 等函数构造优化问题，结合损失与正则求解编辑 `deltas`；
  - 根据超参选择写回方式（原始权重还是 LoRA overlay）。

### 3.2 在本项目中的角色

- **定位**：本项目名中“EMMET Stability Replay”的核心编辑方法主体。
- **代码位置**：
  - 主流程：`src/emmet/emmet_main.py`；
  - 超参定义：`src/emmet/emmet_hparams.py`；
  - 相关扩展：Replay、LoRA-native、Trust 等都围绕该模块实现。
- **如何被调用**：
  - `scripts/run_baseline.py` 中 `--method emmet` 走 EMMET 流程；
  - 常作为“无 Replay / 无 LoRA”的基线，与带 Replay/LoRA 的版本对比。
- **作用**：
  - 提供一个可配置、支持多目标的编辑框架；
  - 为 Memory Replay、LoRA-native、Trust/Rollback 等扩展提供挂载点。

---

## 4. LoRA（Low-Rank Adaptation）

### 4.1 基本原理

- **动机**：直接更新大型权重矩阵非常昂贵且难以管理，希望用更轻量的方式表达“改动”。
- **核心思想**：
  - 不直接改 `W`，而是引入 `ΔW = A B^T`，其中秩 `r` 很小。
  - 训练时只更新 `A, B`，原矩阵 `W` 冻结；
  - 推理时使用 `W' = W + ΔW` 完成新的行为。
- **优点**：
  - 参数量少，训练/编辑更高效；
  - 易于叠加、切换与回滚（只需加载/卸载附加的 LoRA 模块）；
  - 对原始模型破坏更可控。

### 4.2 在本项目中的角色

- **定位**：作为 EMMET 的一种“实现模式”，
  - `edit_mode = "raw"`：直接改真实权重；
  - `edit_mode = "lora_native"`：通过 LoRA overlay 注入编辑。
- **代码位置（结合 todo.md 提示）**：
  - LoRA 后端封装：`src/emmet/peft_backend_lora.py`；
  - LoRA 包装与 bias 前向修复：`src/emmet/lora_wrapper.py`；
  - EMMET 主流程中根据 `edit_mode` 决定使用原始编辑还是 LoRA 编辑：`src/emmet/emmet_main.py`。
- **实验脚本与结果**：
  - 脚本：`scripts/run_lora_native_ablation.cmd/.sh`；
  - 结果汇总：`results/lora_native_ablation.csv`；
  - 图表：`figs/lora_rank_vs_metrics.png`；
  - 文档（计划产出）：`docs/lora_native_editing.md`；
  - 测试（计划产出）：`tests/test_lora_native_backend.py`。
- **作用**：
  - 提供“以 LoRA 形式表达编辑”的对照，与直接改权重（raw）相比，其在稳定性、遗忘、表达能力等方面的差异可通过消融实验观测。

---

## 5. Memory Replay（记忆重放）

### 5.1 基本原理

- **背景**：在持续学习和多轮编辑场景中，后来的更新可能覆盖前面的编辑，导致“灾难性遗忘”。
- **记忆重放的思路**：
  - 维护一个 **历史编辑缓冲区**，存储过去的编辑请求及相关统计；
  - 每当进行新一轮编辑时，从缓冲区采样一部分旧编辑，与当前 batch 一起参与优化；
  - 以此约束新编辑不会完全抹掉旧编辑效果。

### 5.2 本项目中的实现

- **缓冲区结构：`ReplayBuffer`（`src/emmet/replay_buffer.py`）**
  - **`EditRecord`**：封装一次编辑请求，包括：
    - `request`（包含 subject、prompt、target_new 等）；
    - 可选的缓存 keys/values；
    - 时间戳 `timestamp`；
    - 优先级 `priority`；
    - 访问计数 `access_count`（用于 LRU）。
  - **缓冲区策略**：
    - 采样策略：`"random"` / `"priority"` / `"recent"`；
    - 淘汰策略：`"fifo"` / `"lru"`；
    - 去重：按 subject 去重（避免同一 subject 积压过多历史）。

- **合并与数值稳定工具：`src/emmet/replay_utils.py`**
  - `merge_requests`：将当前 batch 与重放样本合并，并分配权重（当前样本权重 1.0，重放样本为 `replay_weight`）；
  - `check_dimension_compatibility`：检查当前 batch 与缓存 keys/values 的维度是否一致；
  - `apply_tikhonov_regularization`：对矩阵加 `λI` 以提升数值稳定性，降低奇异风险；
  - `check_condition_number`：检查矩阵条件数，条件数过高会发出警告；
  - `adaptive_merge_strategy`：根据 buffer 大小与 batch 大小自适应决定重放样本数量；
  - `compute_edit_priority`：根据成功度与重要性系数计算优先级，用于 `priority` 采样策略。

- **主入口：`apply_emmet_with_replay`（`src/emmet/emmet_replay.py`）**
  - **不使用 Replay 时**：
    - 若 `use_replay = False`，则直接调用 `apply_emmet_to_model(...)`，退化为标准 EMMET。
  - **启用 Replay 时**：
    - 从全局 `ReplayBuffer` 按 `replay_rate` 和当前 batch 大小采样若干旧编辑；
    - 用 `merge_requests` 将当前请求与重放样本合并，并记录样本权重与统计信息；
    - 在合并后的请求集合上调用 `apply_emmet_to_model(...)` 完成编辑；
    - 将当前 batch 的每个 request 包装为 `EditRecord`，根据 `compute_edit_priority` 赋予优先级后写回 `ReplayBuffer`；
    - 更新与记录缓冲区的统计信息（大小、唯一 subject 数量等）。

### 5.3 在本项目中的角色

- **定位**：项目名中“Stability Replay”的核心实现，用于提升多轮编辑场景下的稳定性。
- **代码位置**：
  - `src/emmet/emmet_replay.py`；
  - `src/emmet/replay_buffer.py`；
  - `src/emmet/replay_utils.py`。
- **实验与可视化**：
  - 脚本：`scripts/run_replay_ablation.cmd`、`scripts/run_baseline.py` 中与 `--replay_rate`、`--replay_buffer_size`、`--replay_strategy`、`--replay_weight` 等参数相关部分；
  - 可视化工具：`scripts/plot_forgetting_curve.py` 等用于绘制遗忘曲线、旧编辑保留率等指标。
- **作用**：
  - 明确把“旧编辑的保留情况”作为目标之一，通过 Replay 机制减缓灾难性遗忘；
  - 提供一组可控的参数（rate/strategy/buffer size）用于系统性研究稳定性。

---

## 6. 消融实验（Ablation）

### 6.1 基本原理

- **消融实验的目的**：
  - 系统地“关掉或替换”某个组件，观察指标变化；
  - 判断哪些设计真正带来收益，哪些只是复杂度但没有显著贡献。
- **典型操作方式**：
  - 修改某个超参数：例如将 `replay_rate` 从 0 → 0.1 → 0.3 → 0.5；
  - 更换策略：如将 `replay_strategy` 从 `"random"` 换成 `"priority"`；
  - 去掉某个模块：如关闭 LoRA，仅使用 raw 编辑。

### 6.2 在本项目中的实践

- **LoRA 消融**：
  - 脚本：`scripts/run_lora_native_ablation.cmd/.sh`；
  - 内容：系统地遍历 `lora_rank`, `lora_fit_steps` 等超参；
  - 结果：
    - 数据汇总到 `results/lora_native_ablation.csv`；
    - 可视化图如 `figs/lora_rank_vs_metrics.png`；
  - 目标问题：
    - LoRA 维度、训练步数等因素，对 ES/PS/NS 及综合指标的影响如何？

- **Replay 消融**：
  - 脚本：`scripts/run_replay_ablation.cmd`；
  - 内容：改变 `replay_rate`、`buffer_size`、`strategy` 等设置；
  - 目标问题：
    - 引入 Replay 后，在稳定性、遗忘曲线、旧编辑保留率方面的收益多大？

- **综合基线对比**：
  - 脚本：`scripts/run_all_baselines.*`, `run_batch_experiments.py` 等；
  - 将 ROME / MEMIT / EMMET / EMMET+Replay / EMMET+LoRA 等多种方法置于同一评测框架下，比较：
    - ES（编辑成功率）；
    - PS（paraphrase success）；
    - NS（neighborhood specificity）；
    - Composite Score 等综合指标。

- **结果聚合与可视化**：
  - 汇总脚本：`scripts/analyze_results.py`，生成诸如 `results/baseline_comparison/baseline_comparison.csv` 等结果表；
  - 图表示例（见 README 描述）：
    - `efficacy_success_by_method.png`；
    - `paraphrase_success_by_method.png`；
    - `neighborhood_specificity_by_method.png`；
    - `composite_score_by_method.png`；
    - `composite_score_by_batch_size.png`；
    - `composite_vs_batch_size.png`。

---

## 7. 方法之间的整体关系

- **ROME / MEMIT / EMMET**：
  - 三个层次逐渐增强的模型编辑方法：
    - ROME：单编辑，局部 rank-one 线性更新；
    - MEMIT：多编辑联合更新；
    - EMMET：基于 MEMIT 的强化版本，引入更丰富的优化目标与扩展能力。

- **LoRA**：
  - 一种“表达编辑”的方式，将编辑以低秩 overlay 的形式附加到模型上，而非直接改写原始权重；
  - 在本项目中主要通过 `edit_mode = "lora_native"` 与 EMMET 集成。

- **Memory Replay**：
  - 一种多轮编辑场景下的稳定性策略，通过重放历史编辑样本减少灾难性遗忘；
  - 在本项目中以 `ReplayBuffer` + `apply_emmet_with_replay` 的形式实现，围绕 EMMET 展开。

- **消融实验**：
  - 一套评估方法论，用于验证上述设计（特别是 LoRA-native、Replay、Trust/Rollback 等）是否真正有贡献；
  - 通过系统地修改/关掉组件并比较指标，给出有说服力的实验证据。

---

本文件旨在为阅读本仓库代码和实验脚本提供一个“概念地图”。如果需要进一步了解某个方法的实现细节，可以从上述对应的源码路径入手，结合 `scripts/` 目录下的实验入口，顺着调用栈往下看。
