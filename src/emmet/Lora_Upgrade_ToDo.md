# TODO: 基于 LoRA‑Native 的 ΔW 正交化预处理（Orthogonalized Editors）

## 1. 现象与原因：LoRA‑native EMMET 无正交约束下的干扰问题

当前实验设置（`scripts/run_combined_experiments.sh` 中第 3–7 组）采用的是：

- 方法：EMMET + 原生 LoRA（`--edit_mode lora_native --use_lora`）
- Base 模型冻结，所有编辑写入单一 LoRA adapter
- 闭式求解得到每层编辑矩阵 ΔW，并通过 SVD/拟合压缩为 LoRA 的低秩更新

实测现象：

- NS（邻域稳定性）显著提升
- 但 ES（编辑成功率）、PS（paraphrase 成功率）、S（全局 specificity）明显下降

原因（几何视角）：

1. **所有编辑共享同一个低秩 LoRA 子空间**

   - 多条编辑的权重更新被累积到同一组 LoRA 矩阵（同一个 `lora_A`, `lora_B`）。
   - 对于第 k 条编辑，对应的 ΔWₖ 并不是“单独存在”的，而是被压缩进已有子空间。
   - 新的 ΔWₖ 在参数空间中往往与历史 ΔW₁,…,ΔWₖ₋₁ 高度重叠（共享方向）。

2. **低秩压缩 + 子空间重叠导致“交叉投影干扰”**

   - 由于 rank 有限，所有 ΔW 被投射/压缩到一块共享的低维子空间 U。
   - 线性近似下，对某个旧 fact 的输出变化 ≈ 新更新在该 fact 敏感方向上的投影。
   - 当 ΔWₖ 在 U 中的投影与旧编辑所占用的方向高度重合时：
     - 新 LoRA 更新在“旧编辑子空间”上的投影较大；
     - 等价于对旧编辑效果的干扰或部分抵消。

3. **结果表现为：**

   - NS 提升：  
     - Base 冻结 + 低秩 LoRA 使更新相对局部，整体扰动变平滑，因此邻域稳定性提升。
   - ES/PS/S 下降：  
     - 新编辑在共享子空间上的大投影，对旧编辑和相关事实产生“逆向干扰”，
       导致：
       - 已编辑事实效果减弱（ES↓）；
       - 对同一事实的改写提示不稳定（PS↓）；
       - 更广泛知识区域也受到波及（S↓）。

换句话说：当前 LoRA‑native 方案没有对“不同编辑对应的更新子空间”做任何区分或正交约束，导致多个编辑在同一低秩子空间中互相挤压，交叉投影造成干扰。

---

## 2. Orthogonalization 的原理：为何可缓解 ES/PS/S 的下降

核心思想（Orthogonalized Editors）：

> 在 ensemble / PEFT 框架上引入正交化约束，使不同编辑器（或不同 adapter 的更新方向）在参数/激活子空间上尽量正交，从而减少“新编辑对旧编辑的投影干扰”。

几何与线性近似视角：

1. **参数更新视为向量/矩阵，在一个内积空间中相互作用**

   - 把每条编辑的更新视为矩阵 ΔWᵢ，flatten 后可视为向量 wᵢ。
   - 对某个旧 fact 的输出变动，在一阶近似下与参数更新的内积（或投影）成正比。

2. **干扰与投影范数成正比**

   - 对旧编辑 j 来说，新编辑 i 的“干扰量”与新更新在 wⱼ 所在子空间上的投影范数相关：
     - ‖Proj_{span{wⱼ}}(wᵢ)‖ 或更一般的 Proj_{S_old}(wᵢ)。
   - 若强制/鼓励 wᵢ 与已有子空间 S_old 近似正交，投影范数变小，对旧编辑的干扰上界随之收紧。

3. **正交化两种典型做法**

   - 正则项形式：
     - 在拟合/训练 adapter i 的过程中增加项：
       - λ ∑_{j<i} ‖P_{Aⱼ}(Aᵢ)‖² 或 λ ∑_{j<i} ‖Proj_{Sⱼ}(ΔWᵢ)‖²
     - 直接惩罚“新方向在旧子空间上的投影”。
   - 算法形式（Gram‑Schmidt）：
     - 在插入/拟合新 adapter 之前，将其更新方向在旧子空间上做 Gram‑Schmidt 正交化：
       - wᵢ ← wᵢ − Proj_{S_old}(wᵢ)。
     - 等价于显式把新更新推离旧方向。

在我们的 LoRA‑native 场景下，预期效果：

- NS 提升来源（低秩 + base 冻结）保持不变；
- 通过正交化，让新编辑的 ΔW 在“已表达的编辑子空间”上的投影变小：
  - 对旧编辑（ES）和其 paraphrase（PS）的影响减弱；
  - 对更广泛知识区域（S）的无关扰动减小。

因此，从理论和直觉上，Orthogonalization 有潜力在“保住 NS 优势”的同时，缓解 ES/PS/S 的下降。

---

## 3. 基于 LoRA‑Native 的最小改动版：ΔW 正交化预处理方案

目标：在不大改现有 LoRA‑native 实现（`LoRANativeBackend`）的前提下，引入一个**轻量级的 ΔW 正交化预处理**，用于降低编辑之间的交叉投影干扰。

约束条件：

- 仍然使用单一 LoRA adapter（不改为多 adapter 结构）；
- 不改 `LoRANativeBackend.apply_delta` 的接口和内部逻辑；
- 修改集中在 `apply_emmet_to_model` / ΔW 构造阶段，对 ΔW 做处理后再交给 LoRA。

### 3.1 基本思路

1. EMMET 在每层为一批编辑求得原始编辑矩阵集合 {ΔW₁, …, ΔWₙ}；
2. 在**同一层**内，对这些 ΔWᵢ 做一个简单的“正交化预处理”；
   - 例如按编辑顺序或按某种重要性顺序进行；
   - 对每条新的 ΔWᵢ，从已有 ΔW₁,…,ΔWᵢ₋₁ 所张成的子空间上投影扣除；
3. 使用正交化后的 ΔWᵢ' 替换原来的 ΔWᵢ，累积形成该层最终的 ΔW_total'；
4. 将 ΔW_total' 传给 LoRA‑native 的 SVD + 拟合逻辑（`apply_delta`），如同目前的流程。

直觉：

- 先在 ΔW 空间中减少编辑之间的子空间重叠，再做低秩压缩；
- 让 LoRA 的低秩子空间更“分散”，降低新编辑在旧编辑方向上的投影。

### 3.2 具体待办（实现层面）

> 下列条目默认修改位置集中在 `emmet/emmet_main.py` 中构造和累积 ΔW 的部分，必要时可抽一个新工具模块 `emmet/orthogonalization.py`。

1. **梳理当前 ΔW 聚合逻辑**

   - [ ] 在 `apply_emmet_to_model` 中定位：对每条编辑求 ΔW、在每层累积更新的代码段。
   - [ ] 确认每层的 ΔW 是：
     - 先 per‑edit 求解，再求和；还是
     - 直接对整个 batch 求一个 ΔW。
   - [ ] 若是后者（整体 ΔW），需要在 EMMET 求解时保留 per‑edit 或 per‑group 的 ΔW 信息，以便后续正交化。

2. **设计简化版的“同层 ΔW 正交化”接口**

   - [ ] 新增一个函数，例如：
     - `orthogonalize_deltas(layer_deltas: List[Tensor], mode: str = "gs") -> List[Tensor]`
   - [ ] 支持最简模式（Gram‑Schmidt 风格）：
     - 输入：该层所有编辑对应的 ΔWᵢ（flatten 或按矩阵）；
     - 输出：正交化后的 ΔWᵢ'。

   - [ ] 初始实现（伪代码）：

     - 顺序处理：
       - W₁' = W₁
       - 对 i = 2..n：
         - 对 j = 1..i‑1：
           - 计算投影系数：α = ⟨Wᵢ, Wⱼ'⟩ / ⟨Wⱼ', Wⱼ'⟩
           - Wᵢ ← Wᵢ − α Wⱼ'
         - Wᵢ' = Wᵢ

     - ⟨·,·⟩ 可用 Frobenius 内积：⟨A,B⟩ = Tr(AᵀB) = sum(A * B)。

   - [ ] 考虑数值稳定与开关：
     - 对 ‖Wⱼ'‖ 很小的情况跳过；
     - 提供一个开关 `orthogonalization_enabled` / CLI 参数，例如 `--orthogonalize_deltas`。

3. **在 ΔW 聚合前插入正交化预处理**

   - [ ] 在每层 ΔW 求出之后（但在 LoRA `apply_delta` 之前）调用：
     - 若 `orthogonalization_enabled`：
       - 收集该层所有 ΔWᵢ → 调用 `orthogonalize_deltas` → 得 ΔWᵢ'；
       - 使用 ΔWᵢ' 代替原 ΔWᵢ 进行后续累积和 LoRA 拟合。
   - [ ] 若当前实现只保留了“总 ΔW_total”，则需要：
     - 让 EMMET 求解阶段保留 per‑edit ΔWᵢ；
     - ΔW_total' = Σ ΔWᵢ'；
     - 将 ΔW_total' 传入 LoRA。

4. **实验控制与 logging**

   - [ ] 在 `run_baseline.py` 中增加 CLI 参数：
     - `--orthogonalize_deltas`（bool 开关）
     - （可选）`--orthogonalize_mode`，默认 `gs`（Gram‑Schmidt）
   - [ ] 在结果日志中增加以下字段：
     - `orthogonalization_enabled: bool`
     - 层内平均“重叠度”（例如 ΔWᵢ 与 ΔWⱼ 的归一化内积或投影范数）
   - [ ] 在分析脚本中扩展输出：
     - “重叠度 vs 干扰”曲线：
       - x 轴：平均/最大重叠度（投影范数比例）；
       - y 轴：ES、PS、NS、S 等指标。

5. **效率与简化方案**

   - [ ] 若 per‑edit 粒度较细导致正交化开销过大，可先尝试：
     - 按“小批编辑 group”进行正交化（例如按 relation/实体分组）；
     - 或仅对“干扰敏感层”（如中间若干 transformer block）进行正交化。
   - [ ] 若 full Gram‑Schmidt 过慢，可尝试：
     - 仅对“新批次 ΔW_new”在“历史 ΔW_old 子空间”上做一次投影扣除，而非对同批次内部全部正交化（online 版本）。

---

## 4. 预期结果（Hypothesis）

在现有 LoRA‑native 设置基础上，引入上述 ΔW 正交化预处理后，预期：

1. **NS（邻域稳定性）**  
   - 依然优于纯 EMMET baseline；
   - 相比“无正交化”的 LoRA‑native，变化可能不大或略有提升（因更新方向更分散）。

2. **ES / PS（编辑成功与 paraphrase 成功）**  
   - 相比“无正交化”的 LoRA‑native，明显改善：
     - 新编辑对旧编辑的逆向干扰减弱；
     - paraphrase 场景中的表现更接近纯 EMMET baseline。

3. **S（global specificity / 后向遗忘）**  
   - 对不相关知识的扰动减少：
     - 在“重叠度 vs 干扰”分析中，随着 ΔW 子空间重叠度下降，S 指标的下降幅度减小；
     - 在合理的正交化强度下，有望接近或稍优于纯 EMMET baseline。

4. **整体权衡**  
   - 在“无 LoRA baseline”与“无正交约束 LoRA‑native”之间，期望找到一个折中点：
     - NS 保留 LoRA 优势；
     - ES/PS/S 介于两者之间，显著优于当前“NS↑ 但 ES/PS/S 明显下降”的状态。

简言之：ΔW 正交化预处理希望实现——**利用 LoRA 的低秩与冻结优势提升 NS，同时通过几何上的正交化减少编辑之间的投影干扰，从而缓解 ES/PS/S 的退化**。