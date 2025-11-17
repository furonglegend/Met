# Greedy LoRA + Online ES/PS/NS Improvement ToDo
python scripts/run_baseline.py --method emmet --model gpt2-xl --dataset counterfact_500 --num_edits 20 --batch_size 4 --edit_mode lora_native --lora_rank 16 --trust_enable --greedy_lora_steps 4 --greedy_lora_min_improvement 0.02 --greedy_lora_patience 1 --output_dir results/debug_greedy_online
> 目标：让 `greedy_lora_min_improvement` 真正变成“在线指标改进门槛”，即：
> 在 LoRA-native + Greedy 路径中，每一步小步更新后，对一个小批 probe prompts 在线计算 ES/PS/NS（或 composite）；
> 如果当前步的指标相对上一状态没有达到最小改进量，就**不接受**这一步（回滚这一步的 LoRA 更新），并根据 patience 提前终止 greedy。

---

## Phase 1：接口与抽象设计

- [ ] **确定在线评估的生效范围与接口位置**
  - [ ] 仅在 `edit_mode == "lora_native"` 且 `greedy_lora_steps > 1` 时启用在线评估；
  - [ ] 决定在线 ES/PS/NS 的计算代码放在哪一层：
    - 方案 A（推荐，侵入性较小）：在 `src/emmet/emmet_main.py` 中的 `apply_emmet_to_model` LoRA-native greedy loop 内部，调用一个“轻量评估回调”；
    - 方案 B：在 `scripts/run_baseline.py` 的 batch 循环中维护“在线 evaluator”，但会让 greedy 逻辑分散到脚本层，不够内聚。
  - [ ] 最终建议：
    - 在 `src/emmet/emmet_main.py` 中新增一个小型工具函数（例如 `quick_eval_es_ps_ns`），依赖一个注入进来的“预测函数”来复用 `run_baseline.py` 中 `_test_prediction` 的逻辑。

- [ ] **定义一个“在线评估回调”抽象（可选）**
  - [ ] 在 `apply_emmet_to_model` 的参数列表中，新增一个可选回调参数，例如：

    ```python
    def apply_emmet_to_model(...,
                             metric_fn: Optional[Callable[[AutoModelForCausalLM, AutoTokenizer, List[Dict]], Dict[str, float]]] = None,
                             ...):
    ```

  - [ ] 语义：
    - 输入：当前模型 `model`、`tok`、一个小 batch 的 `probe_requests`；
    - 输出：形如 `{"es": float, "ps": float, "ns": float, "composite": float}` 的字典；
    - 若 `metric_fn is None`，则不启用在线 greedy 评估，`greedy_lora_min_improvement` 退化为占位（与当前行为兼容）。

- [ ] **在 `scripts/run_baseline.py` 中接好回调**
  - [ ] 在 `BaselineRunner.run_editing` 中，构造一个简单闭包 `metric_fn`，内部调用现有 `_test_prediction`：

    ```python
    def make_metric_fn(logger, sequence_eval_limit: int = 8):
        def metric_fn(model, tok, probe_requests: List[Dict]) -> Dict[str, float]:
            # 选取少量 probe（例如 4~8 条）
            # 复用 _test_prediction 计算 ES/PS/NS
            # 返回 {"es": ..., "ps": ..., "ns": ..., "composite": ...}
        return metric_fn
    ```

  - [ ] 调用 `apply_emmet_to_model` 时，把 `metric_fn` 通过关键字参数传入（只在 `edit_mode=="lora_native"` 时传）。

---

## Phase 2：在线 ES/PS/NS 快速评估实现

> 实现一个“小而快”的在线评估，只对少数 probe 做 forward，避免大幅拖慢训练。

- [ ] **确定 probe 采样策略**
  - [ ] 基本策略：从当前 `batch_requests` 中采样一小批（例如 4 条）作为 probe；
  - [ ] 选取每条 request 的以下 prompts：
    - rewrite prompt：`request["prompt"].format(subject)` → ES；
    - 1~2 个 paraphrase prompts（若存在）→ PS；
    - 1~2 个 neighborhood prompts（若存在）→ NS。
  - [ ] 限制总测试条数，比如：
    - 总 ES 测试 ≤ 8 条，PS/NS 各 ≤ 8 条，以控制推理成本。

- [ ] **复用 `_test_prediction` 逻辑**
  - [ ] 将 `BaselineRunner._test_prediction` 中的核心逻辑提升为一个可重用函数（避免重复代码）：

    - 方案 A：在 `scripts/run_baseline.py` 顶部定义一个模块级函数 `_test_prediction_core(model, tokenizer, prompt, target_new, target_true)`，既供 `BaselineRunner` 使用，也供 `metric_fn` 调用；
    - 方案 B：在 `src/utils` 下新建一个小工具模块（例如 `src/utils/eval_helpers.py`），放置该函数，然后两边统一 import。

  - [ ] 在 `metric_fn` 内部，循环 probe，调用 `_test_prediction_core` 计算：

    ```python
    es_list, ps_list, ns_list = [], [], []
    # ES：rewrite prompts
    # PS：paraphrase_prompts
    # NS：neighborhood_prompts
    composite = (es_mean + ps_mean + ns_mean) / 3.0
    return {"es": es_mean, "ps": ps_mean, "ns": ns_mean, "composite": composite}
    ```

- [ ] **性能注意事项**
  - [ ] 避免在 greedy 每一步都用完整 batch 做评估，只用固定上限的小批；
  - [ ] 设置一个 config 项（例如 `greedy_online_eval_max_probes`），未来方便调节数量；
  - [ ] 允许通过 CLI 参数关闭在线 greedy 评估（例如 `--greedy_lora_use_distance_only`），直接回退到 distance-based trust。

---

## Phase 3：在 LoRA-native Greedy Loop 中接入接受判据

> 修改 `src/emmet/emmet_main.py` 中的 LoRA-native 分支，让 `greedy_lora_min_improvement` 真正控制“接受/拒绝某一步”。

- [ ] **定位并理解现有 Greedy 逻辑**
  - [ ] 文件：`src/emmet/emmet_main.py`
  - [ ] 函数：`apply_emmet_to_model`
  - [ ] LoRA-native 分支（简化伪代码）：

    ```python
    if use_lora_native:
        base_norm = torch.norm(base_upd_matrix)
        cumulative_delta_norm = 0.0
        patience_counter = 0
        last_accepted_step = -1

        for greedy_step in range(greedy_steps):
            step_scale = 1.0 / float(greedy_steps)
            upd_matrix = base_upd_matrix * step_scale

            # step=0 trust gate
            # 调用 lora_backend.apply_delta(...)
            # 更新 cumulative_delta_norm / last_accepted_step / effective_upd_matrix
            # 检查 base_norm guard
            # 检查 greedy_patience 作为简单 cap
    ```

- [ ] **增加“基线指标”和“当前指标”的维护变量**
  - [ ] 在进入 greedy loop 前：

    ```python
    best_metrics = None
    last_composite = None
    if metric_fn is not None and len(requests) > 0:
        # 评估当前 LoRA 状态的 ES/PS/NS（在应用任何 greedy 步之前）
        probe_requests = sample_probe_requests(requests, max_probes=4)
        best_metrics = metric_fn(model, tok, probe_requests)
        last_composite = best_metrics.get("composite", None)
    ```

  - [ ] 该 `last_composite` 视为 greedy 的“起点指标”，后续每一步都需要相对于它做比较。

- [ ] **在每一步应用完 LoRA delta 后做在线评估**
  - [ ] 在 `lora_backend.apply_delta(...)` 成功之后，如果 `metric_fn` 可用，则：

    ```python
    if metric_fn is not None and last_composite is not None:
        current_metrics = metric_fn(model, tok, probe_requests)
        curr_comp = current_metrics.get("composite", None)
        if curr_comp is not None:
            improvement = curr_comp - last_composite
            if improvement >= greedy_min_improve:
                # 接受：更新 best_metrics / last_composite
                best_metrics = current_metrics
                last_composite = curr_comp
                last_accepted_step = greedy_step
                greedy_reason = "metric_improved"
            else:
                # 不接受：回滚本步 LoRA 更新
                rollback_last_lora_step(lora_backend, w_name)
                patience_counter += 1
                greedy_reason = "no_improvement"
                if greedy_patience > 0 and patience_counter >= greedy_patience:
                    break
                # 继续下一步（不更新 last_composite）
    ```

  - [ ] 关键点：需要一个“回滚本步 LoRA 更新”的机制。

- [ ] **实现 LoRA 单步回滚机制**
  - [ ] 当前 `LoRANativeBackend` 没有内建“undo 上一步”的接口，只能：
    - 方案 A：在每次 greedy 开始前保存 LoRA 层的初始参数快照，之后若某一步被拒绝，就用快照恢复，然后只累积“被接受的步”；
    - 方案 B：在每一步前保存当前 LoRA 层参数，评估后若拒绝则立即恢复；
  - [ ] 推荐方案 B（逻辑清晰，但显存占用略大）：

    - 在 LoRA-native 分支中，每次循环的开头：

      ```python
      state_before_step = lora_backend.snapshot_layer_state(w_name)  # 需要新增接口
      ```

    - 若该步最终被拒绝：

      ```python
      lora_backend.restore_layer_state(w_name, state_before_step)
      ```

    - 在 `LoRANativeBackend` 中新增：

      ```python
      def snapshot_layer_state(self, weight_param_name: str) -> Dict[str, torch.Tensor]:
          module_path = weight_param_name[:-7]
          layer = self._ensure_lora_layer(module_path)
          return {
              "A": layer.lora_A.detach().clone(),
              "B": layer.lora_B.detach().clone(),
          }

      def restore_layer_state(self, weight_param_name: str, state: Dict[str, torch.Tensor]) -> None:
          module_path = weight_param_name[:-7]
          layer = self._ensure_lora_layer(module_path)
          layer.lora_A.data.copy_(state["A"])
          layer.lora_B.data.copy_(state["B"])
      ```

- [ ] **将 `greedy_lora_min_improvement` 接入逻辑**
  - [ ] 解释 `greedy_lora_min_improvement`：以“composite 的绝对提升量”作为单位，例如 0.01 表示提升 1 个百分点：

    $$
    \Delta S = S_{\text{current}} - S_{\text{last}} \geq \text{greedy\_lora\_min\_improvement}
    $$

  - [ ] 在代码中直接用 `improvement >= hparams.greedy_lora_min_improvement` 作为接受条件；
  - [ ] 如果 `metric_fn is None` 或 `last_composite is None`，则自动接受所有步（退化为旧行为）。

- [ ] **Greedy 事件记录中加入原因字段**
  - [ ] 在构造 `distances[layer]` 的 `temp_dict` 时：

    ```python
    temp_dict['greedy_step'] = max(0, last_accepted_step)
    temp_dict['greedy_accept'] = (effective_upd_matrix is not None)
    temp_dict['greedy_scale'] = step_scale or ...
    temp_dict['greedy_reason'] = greedy_reason  # 'metric_improved', 'no_improvement', 'trust_low', ...
    ```

  - [ ] 这样 `scripts/run_baseline.py` 已经会把 `greedy_reason` 写入 `trust_events.jsonl`，便于后续分析。

---

## Phase 4：CLI 与配置层面的小扩展

- [ ] 在 `EMMETHyperParams` 中：
  - [ ] 保持现有字段不变：`greedy_lora_steps`, `greedy_lora_min_improvement`, `greedy_lora_patience`, `greedy_lora_use_distance_only`; 
  - [ ] 在文档或注释中明确：
    - 当 `greedy_lora_use_distance_only=True` 时：只使用 trust/distance 作为 gate，`greedy_lora_min_improvement` 不启用在线 metrics；
    - 当 `greedy_lora_use_distance_only=False` 时：若 `metric_fn` 不为空，则用在线 composite 作为接受判据。

- [ ] 在 `scripts/run_baseline.py` CLI 中（可选）：
  - [ ] 增加一个参数：

    ```python
    parser.add_argument("--greedy_lora_use_distance_only", action="store_true", default=True,
                        help="If set, greedy acceptance only uses distance/trust, not online ES/PS/NS.")
    ```

  - [ ] 或者复用已有字段，直接在 `EMMETHyperParams` JSON 中配置。

---

## Phase 5：验证与调参与注意事项

- [ ] **单批小实验验证**
  - [ ] 选择一个非常小的设置（例如 `num_edits=20`, `batch_size=4`）：

    ```powershell
    python scripts/run_baseline.py `
      --method emmet `
      --model gpt2 `
      --dataset counterfact_500 `
      --num_edits 20 `
      --batch_size 4 `
      --edit_mode lora_native `
      --lora_rank 16 `
      --trust_enable `
      --greedy_lora_steps 4 `
      --greedy_lora_min_improvement 0.02 `
      --greedy_lora_patience 1 `
      --output_dir results/debug_greedy_online
    ```

  - [ ] 检查 `trust_events.jsonl` 中：
    - `greedy_step`、`greedy_accept`、`greedy_reason` 是否与预期一致（如有大量 `no_improvement`，说明门槛可能偏高）。

- [ ] **调参建议**
  - [ ] 起始建议：
    - `greedy_lora_steps ∈ {3,4}`；
    - `greedy_lora_min_improvement ∈ {0.01, 0.02}`；
    - `greedy_lora_patience ∈ {1,2}`；
  - [ ] 观察 ES/PS/NS 的变化：
    - 如果 ES/PS 提升，但 NS 降得多，适当提高 trust_threshold 或降低 greedy 步数；
    - 如果几乎没有步被接受，降低 `greedy_lora_min_improvement` 或增加 patience。

- [ ] **性能与稳定性**
  - [ ] 注意在线评估是额外的 forward 成本，大规模实验前先在小规模上测好时间开销；
  - [ ] 避免每层、每步都用不同 probe，先固定一组 probe_requests 复用整个 greedy 序列，以减少方差和计算量；
  - [ ] 如遇到显存问题，可以考虑：
    - 降低 `greedy_lora_steps`；
    - 减少 `max_probes`；
    - 只在部分层上启用 greedy（例如高层）。

---

## 小结

这一 ToDo 文件的目标是把目前“占位”的 `greedy_lora_min_improvement` 变成真正的“在线指标改进门槛”，其关键工作包括：

1. 设计并接入一个轻量的在线 ES/PS/NS 评估函数（复用 `_test_prediction`）；
2. 在 LoRA-native greedy loop 中，在每一步小步更新后调用在线评估，并根据 `Δcomposite ≥ greedy_lora_min_improvement` 决定是否接受该步；
3. 为单步添加回滚机制，保证“不接受的步”不会残留在 LoRA 因子中；
4. 将 accept/reject 原因写入 `trust_events.jsonl`，用于后续分析 greedy 行为与最终指标的关系。

完成以上步骤后，你就可以真正做到：“如果每一步在线 ES/PS/NS 上没有 improvement，就不接受这一步”，并通过 `greedy_lora_min_improvement` 在 CLI 上直接控制这一门槛。