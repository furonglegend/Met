# Greedy LoRA Incremental Editing ToDo（基于 EMMET + LoRA + Trust）

> 目标：在当前 EMMET + LoRA-native 框架上，实现“基于 Trust 的贪心式多步低秩编辑”，即：
> 对同一 sample，在已有 LoRA 编辑结果基础上，迭代执行小步低秩更新；每一步根据 Trust 与指标改进决定是否保留；
> 期望在 **不显著降低 NS** 的前提下，逐步提升 ES/PS/composite，使最终结果优于单次 EMMET 编辑。

---

## Phase A：现状审查与需求精炼

- [ ] **梳理现有实现**
  - [ ] 阅读 `src/emmet/emmet_main.py` 中 `execute_emmet` 与 `apply_emmet_to_model` 的调用关系，确认：
    - 闭式解求出的 `deltas` 只计算一次（对一个 `requests` 批）；
    - 当前 LoRA-native 路径仅在 `apply_emmet_to_model` 中执行一次 `lora_backend.apply_delta(...)`。
  - [ ] 阅读 `src/emmet/peft_backend_lora.py`，确认：
    - LoRA 参数的注入方式（`LoRALayer` 的结构、`apply_delta` 的行为）；
    - 是否支持重复对同一层多次调用 `apply_delta` 并累积更新；
    - Fallback 与 residual guard 的触发点。
  - [ ] 阅读 `src/emmet/trust.py` + `apply_emmet_to_model` 中的 trust 逻辑，确认：
    - 当前 `trust_score` 是基于 `preservation_distance/new_edit_distance/old_edit_distance` 的闭式 gating；
    - 当 `trust_score < trust_threshold` 时是“跳过/缩放本次更新”，仅针对**单次** delta。
  - [ ] 检查 `scripts/run_baseline.py` 中 `edit_mode="lora_native"` + `trust_*` 参数与 hparams 的映射是否一致。

- [ ] **明确 Greedy LoRA 的目标行为**
  - [ ] 对同一条 sample / 同一批 `requests`：
    - 在一次闭式解基础上，允许多轮低秩“小步编辑”；
    - 每轮以“当前 LoRA 权重作为起点”，产生一个新的 candidate delta；
    - 用 Trust 评估该小步是否保留（accepted）或回滚（rejected）；
    - 若保留，则下一轮在 **已保留的 LoRA 结果** 上继续；
    - 若连续若干轮无提升，可提前停止。
  - [ ] 这与现有 `replay_rate` 的“跨样本 memory replay”不同：
    - Greedy LoRA 聚焦于“**同一 sample 的多次尝试**”；
    - Memory Replay 聚焦于“历史样本的再利用”。

---

## Phase B：接口与配置设计

### B1. 新增/扩展超参数与配置

- [ ] 在 `src/emmet/emmet_hparams.py` (`EMMETHyperParams`) 中新增字段：
  - [ ] `greedy_lora_steps: int` — 对同一 batch / 同一条请求允许的最大 Greedy 步数（默认 1 = 现有行为）；
  - [ ] `greedy_lora_min_improvement: float` — 单步视为“有效提升”的最小增益门槛（例如 composite 提升 > 0.01）；
  - [ ] `greedy_lora_use_distance_only: bool` — 是否仅依赖 distance-based trust（默认 True），后续可扩展为结合 ES/PS 等；
  - [ ] `greedy_lora_patience: int` — 连续多少步无提升则提前停止（如 2 或 3）。

- [ ] 在 `scripts/run_baseline.py` CLI 中增加对应参数，并写入 `ExperimentConfig` → `EMMETHyperParams`：
  - [ ] `--greedy-lora-steps`（映射到 `hparams.greedy_lora_steps`）；
  - [ ] `--greedy-lora-min-improvement`；
  - [ ] `--greedy-lora-patience`；
  - [ ] 仅当 `--edit_mode lora_native` 时才生效，其他模式忽略/警告。

- [ ] 确认/更新 hparams JSON（例如 `src/hparams/EMMET/gpt2*.json`）：
  - [ ] 为上述字段添加合理默认值，确保不配置时行为等价于当前实现。

### B2. Trust 与 Greedy 的结合策略

- [ ] 定义一个“单步是否接受”的判定函数（伪逻辑）：
  - 输入：
    - 当前步的 `trust_score_step`；
    - 与上一状态相比的指标增益/距离改进（如果可用：ES/PS/NS/composite 或 distance-based gain）；
    - 全局阈值：`trust_threshold`、`greedy_lora_min_improvement`。
  - 输出：
    - `accept: bool`；
    - `reason: str`（如 `"trust_low"`, `"no_improvement"`, `"improved"`）。
  - [ ] 初版可采用简单策略：
    - `trust_score_step is None` → 直接拒绝或只接受非常小步（视配置而定）；
    - 若 `trust_score_step < trust_threshold` → 拒绝该步；
    - 否则若 distance gain < `greedy_lora_min_improvement` → 视为无提升，可记入 patience 计数；
    - 同时若可计算“distance gain”（如 new_edit_distance 降低量），可作为改进信号。

---

## Phase C：核心算法改造（Greedy 多步 LoRA 编辑）

> 目标文件：`src/emmet/emmet_main.py`（`apply_emmet_to_model` + 必要时 `execute_emmet`）

### C1. 保持单次闭式解不变

- [ ] 保持 `execute_emmet` 的接口与行为 **不做结构性修改**：
  - 仍然对给定 `requests` 和 `hparams.layers` 计算一次 `deltas`：
    - 每一层对应一个 `(key_mat, val_mat, preservation_distance, new_edit_distance, old_edit_distance, inside_norms)`；
  - 这一步可以视作“基线 delta 提案生成器”。

- [ ] Greedy 多步逻辑应置于 `apply_emmet_to_model`，在已有 delta 基础上迭代调用 LoRA backend。

### C2. 在 LoRA-native 路径上增加外层 Greedy loop

- [ ] 针对 `use_lora_native` 分支，重构如下结构（伪代码）：

  ```python
  use_lora_native = (hparams.edit_mode == "lora_native")
  greedy_steps = max(1, getattr(hparams, "greedy_lora_steps", 1))
  
  # 单次调用 execute_emmet，得到基线 deltas（保持现状）
  deltas = execute_emmet(...)
  
  # LoRA backend 初始化（保持现状）
  lora_backend = LoRANativeBackend(...)
  
  # Greedy 多步循环
  for step in range(greedy_steps):
      # 1) 基于 deltas 产生本步的 candidate upd_matrix
      #    - 可以：
      #      a) 每步重复使用同一个 deltas（但缩小 scale / 调整 alpha）；或
      #      b) 后续扩展为根据当前状态重新估计 deltas（高阶版本，初版可以先用 a）
      
      # 2) 对每个 weight_name 计算 trust_score_step，并根据 trust_threshold/Greedy 规则决定：
      #    - 接受：调用 lora_backend.apply_delta(...)
      #    - 拒绝：本步对该 weight_name 不应用 delta
      
      # 3) 如果本步没有任何 layer 被接受，增加 patience 计数；
      #    - 若连续 patience 步无接受，提前终止 Greedy loop。
  ```

- [ ] 初版实现建议：**不重新调用 `execute_emmet`**，而是：
  - 使用同一个 `upd_matrix`，但在多步中对其施加递减 scale：

    $$
    \Delta W^{(\text{step})} = \eta_{\text{step}} \cdot \Delta W_{\text{base}}, \quad \eta_{\text{step}} \in (0,1]
    $$

  - 例如：`eta_step = base_scale * decay_factor**step`；
  - Trust 负责在每一步 gating：步子虽小，但如果已明显损伤邻域，仍可被拒绝。

### C3. 每步的 Trust / 距离度量获取

- [ ] 当前 `execute_emmet` 只计算一次 `preservation_distance/new_edit_distance/old_edit_distance`，与“初始 delta 应用前后”绑定。
  - 若要在 Greedy 多步中对“累积后的 LoRA 状态”进行评估，需要新增一种
    **“在线/近似 distance 估算”机制**。

- [ ] 为降低复杂度，初版可采用折中方案：
  - 把 `compute_trust_score` 当作 **“初始候选 delta 的静态信任度”**：
    - 如果初始 trust 很低（`< thr_low`），则完全不进入 Greedy 多步（直接回滚/缩放）；
    - 如果初始 trust 中等或较高，则允许多步，但步长可以依据 trust 大小调整。
  - 在这个简化版本中，Greedy loop 的每步 trust 可以“复用初始 trust_score”或采用非常轻量的启发式（如 delta_norm 累积阈值）。

- [ ] 若后续有时间，可以在 Greedy 路径中加入：
  - 轻量的 held-out 评估（选少量样本快速测 ES/NS 改变）；
  - 基于 `calculate_distances` 的近似版本，用当前 LoRA 状态重算 distance（代价较高，但更精确）。

### C4. 记录 Greedy 步级别的事件（trust_events 扩展）

- [ ] 在 `apply_emmet_to_model` 组装 `distances` 时，为每个 layer 的字典添加 Greedy 相关字段：
  - [ ] `greedy_step`: 当前步编号（0,1,2,...）；
  - [ ] `greedy_accept`: 是否在该步被接受（True/False）；
  - [ ] `greedy_scale`: 本步使用的 `eta_step`；
  - [ ] `greedy_reason`: 接受/拒绝原因（例如 `"trust_low"`, `"no_improvement"` 等）。

- [ ] 确保 `scripts/run_baseline.py` 中写出的 `trust_events.jsonl` 同样记录这些字段，以便后续分析：
  - 在已有的 `tev = {"run_dir", "batch_idx", "layer", ...}` 基础上附加 `greedy_step/greedy_accept/...`。

---

## Phase D：评估 Greedy LoRA 的效果

### D1. 实验设置与对照组

- [ ] 定义实验配置（以 GPT-2 / CounterFact-500 为起点）：
  - [ ] Baseline 1：EMMET + raw（无 LoRA，无 Greedy，无 Replay）
  - [ ] Baseline 2：EMMET + LoRA-native（`greedy_lora_steps=1`）
  - [ ] Greedy LoRA：EMMET + LoRA-native + Greedy（`greedy_lora_steps ∈ {3,5}`）

- [ ] 可选：与 Memory Replay 组合：
  - [ ] EMMET + Replay + LoRA-native（`greedy_lora_steps=1`）
  - [ ] EMMET + Replay + LoRA-native + Greedy

### D2. 指标与分析

- [ ] 对比以下指标（使用 `scripts/analyze_results.py`）：
  - [ ] `efficacy_success` / `paraphrase_success` / `neighborhood_specificity` / `composite_score`；
  - [ ] Greedy 步数 vs 平均指标变化；
  - [ ] 历史编辑保留率（可以近似为：在长序列编辑中已有事实仍被正确预测的比例）。

- [ ] 分析 `trust_events`：
  - [ ] 每个 layer 的 `greedy_accept` 率；
  - [ ] `greedy_step` 分布：多数提升发生在哪几步；
  - [ ] 信任度（trust_score）与最终 metrics 改善之间的相关性。

### D3. 预期结果与检查项

- [ ] 在 **相似或略高的 NS** 下，Greedy LoRA 的 ES/PS/composite 相比单步 LoRA 有一定提升；
- [ ] Greedy 步数不宜过大：超过某个步数后收益递减甚至恶化（可据此选择默认 `greedy_lora_steps`）。

---

## Phase E：进一步的精炼与扩展（可选）

- [ ] 基于真实 ES/PS/NS 的在线评估：
  - 对每个 Greedy 步抽取小批 probe prompts，估计该步对目标/邻域的影响，将其纳入 `is_better` 判据；
  - 形成一个更完整的 TrustMetrics（不局限于 distance）。

- [ ] 针对“低秩表达能力不足”的样本做特例处理：
  - 在连续多步均无有效提升时，触发“放宽约束”：
    - 临时提高 LoRA rank；或
    - 对个别层退回 raw 编辑（若允许）。

- [ ] 与 Memory Replay 的双层策略：
  - 外层：跨样本的 Memory Replay，保证旧事实不被遗忘；
  - 内层：对单一事实的 Greedy LoRA，提升该事实的局部表现。

---

## 小结

该 ToDo 计划的落脚点是：

1. **不动 `execute_emmet` 的核心闭式解逻辑**，仅在 `apply_emmet_to_model` 的 LoRA-native 路径上增加 Greedy 多步控制；
2. 通过 **浅层 trust gating + 步长衰减** 尝试在 low-rank 子空间内做贪心式局部优化；
3. 先实现一个依赖 distance-based trust 的轻量版本，再视时间将 ES/PS/NS 引入单步接受判据；
4. 所有决策事件写入 `trust_events.jsonl`，便于之后分析 Greedy 的真实贡献与失败模式。

完成以上任务后，你就有了一个与当前 Replay/Trust 体系兼容的、可控的 “Greedy LoRA 多步低秩编辑” 实验管线，可以直接纳入现有的结果分析与可视化框架。