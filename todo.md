# 项目路线图与待办清单（基于 EMMET 与统一编辑框架）

> 目标：在现有 `unified-model-editing` 代码与报告基础上，完成“可复现的基线 + 稳定性增强（Memory Replay + PEFT）+ 系统化评测与分析”，并于截止日期前形成完整实验与写作成果。

## 1. 核心目标与验收标准

- [ ] 复现 ROME / MEMIT / EMMET 在小规模 CounterFact 子集上的编辑结果（含 ES/PS/NS/GE/S 指标）。
- [ ] 实现 EMMET + Memory Replay 机制，显著减缓序列/批量编辑后的遗忘（优先关注 NS/历史编辑保留）。
- [ ] 实现 EMMET + PEFT（LoRA/Adapters）版本的编辑，限制参数更新子空间以提升稳定性。
- [ ] 在不同编辑批量与顺序（单次批量、序列批量）下做系统对比，完成可复现实验脚本与可视化。
- [ ] 输出一份包含方法、实验与结论的最终报告/笔记本与可运行脚本。

验收产物：

- 实验脚本与配置（可一键运行、含随机种子与日志）。
- 结果表格与图：ES/PS/NS/GE/S、遗忘曲线、批量规模对性能的影响、顺序-批量对比。
- 简要技术报告草稿（可合入最终报告）。

## 2. 需要学习/补齐的知识点

理论/算法：

- [ ] ROME 等式约束编辑的基本推导与目标函数；MEMIT 的最小二乘放宽、批量编辑思想。
- [ ] EMMET 的统一视角与闭式解（等式约束的批量解法），理解 KKT 条件、约束最小二乘与矩阵分解的数值稳定性问题。
- [ ] 遗忘现象：渐进遗忘与灾难遗忘的区别与成因；顺序编辑与批量编辑的干扰机制。

工程/实践：

- [ ] `unified-model-editing` 代码结构：数据集管线（CounterFact/ZSRE）、`emmet/memit/rome` 各模块、`util/nethook.py` 钩子机制、评测脚本与指标实现。
- [ ] PyTorch Hook/Forward Intervene 技巧、Hugging Face Transformers（以 Llama 系列为例）的层命名与权重注入位置。
- [ ] PEFT（LoRA/Prefix/Adapters）基本原理与库用法（优先 LoRA），如何将更新限制在低秩/适配器参数中。
- [ ] 实验可复现性：随机种子、设备差异、批量大小与显存管理、日志（建议本地 CSV/JSON，云上可选）。

## 3. 环境与数据准备（优先完成）

- [x] 使用 conda 创建环境并安装依赖（优先 `conda`，必要时再补 `pip`）。
- [ ] 校验 Python 版本与 PyTorch/CUDA，验证能加载测试模型与运行一个最小推理脚本。
- [ ] 准备数据集：`data/counterfact_*.json` 已提供子集，验证读取与采样脚本 `create_samples_cf.py` 可用。
- [ ] 准备基础模型权重（小模型优先，如 Llama-2/3-Instruct 小尺寸或 GPT2 作为快速迭代备选）。

预期产出：`env.yaml`/`requirements.txt`（标注使用 conda 优先安装），一段最小可运行的模型加载与推理验证日志。

## 4. 复现实验基线（ROME / MEMIT / EMMET）

- [ ] 选取 CounterFact 的一个小子集（如 200～500 条）用于快速验证与调参。
- [ ] 运行 ROME/MEMIT/EMMET 的最小示例（参考 `rome_main.py`、`memit_main.py`、`emmet_main.py` 与 `hparams/*`）。
- [ ] 使用提供的评测脚本：`downstream_eval/current_edit_performance.py`、`downstream_eval/current_edit_scores.py`、`glue_eval/*`（可先跳过 GLUE，待方法稳后再跑）。
- [ ] 记录 ES/PS/NS/GE/S 指标，并保存到本地 `results/*.csv`。

预期产出：一个 `scripts/run_baselines.cmd` 或 `.sh`（如在 WSL/Unix 环境）与 `results/baselines_{date}.csv`。

## 5. 实现 Memory Replay for EMMET（稳定性增强 1）

设计要点：

- [ ] Buffer 结构：存储过往编辑的 (subject, relation, object)、原/改写提示、必要的中间统计（若可复用，如 Keys/Values/层位点）。
- [ ] 采样策略：每次新编辑时，按比例 r 从 Buffer 采样历史样本并一并纳入 EMMET 的等式约束/矩阵求解中。
- [ ] 代价与稳定性：控制历史约束权重或数量，避免矩阵病态；必要时加入正则或数值技巧（加噪、Tikhonov 等）。
- [ ] 配置化：在 `hparams/EMMET` 增加 `replay_rate`、`replay_max_items`、`replay_weight` 等参数；日志记录实际使用的历史样本数与指标变化。

实施步骤：

- [ ] 阅读 `emmet/compute_ks.py`、`emmet/compute_z.py` 与 `emmet_main.py`，定位等式约束构建与闭式解步骤。
- [ ] 在构建约束时拼接当前批与历史批（按采样策略），确保维度与索引一致。
- [ ] 增加缓冲区维护模块（插入、采样、去重策略），并在成功编辑后更新 Buffer。
- [ ] 单元测试：小批量 5～20 条，验证解能收敛、指标不回退（尤其 NS）或回退可控。

预期产出：`emmet/replay_buffer.py`、`emmet/emmet_main.py` 中的可选 `--use_replay` 逻辑与新增 hparams；`results/replay_ablation.csv`。

## 6. 实现 EMMET + PEFT（LoRA/Adapters）（稳定性增强 2）

设计要点：

- [ ] 在目标层仅更新 LoRA/Adapter 参数，冻结原权重；等式约束/闭式解仅作用于可训练（低秩）参数。
- [ ] 可选两种路径：
  - 直接用 Hugging Face PEFT 库封装目标层；
  - 自实现低秩分解（W ≈ W0 + A·B），将闭式解作用到 A/B 的梯度/参数更新上。
- [ ] 配置化 LoRA 超参：rank、alpha、dropout；与 Replay 组合时要控制总约束规模与求解稳定性。

实施步骤：

- [ ] 为目标 MLP/Attention 层添加 LoRA 适配模块与开关（`--use_lora`）。
- [ ] 在求解/更新路径中只触达 LoRA 参数；验证参数量与显存占用的下降。
- [ ] 结合 Replay 做小规模对比实验，观察 NS/历史编辑保留的变化。

预期产出：`emmet/lora_layers.py` 或基于 PEFT 的封装、更新的 `hparams/EMMET`、`results/peft_ablation.csv`。

## 7. 实验设计与运行矩阵

- [ ] 批量规模：{1, 8, 32, 128, 512, 1024}；
- [ ] 顺序 vs 批量：仅顺序单条；分段序列-批量（如每批 32，连续 32 批，共 1024 条）；
- [ ] Replay 比例 r：{0, 0.1, 0.3, 0.5}；
- [ ] LoRA rank：{4, 8, 16}；
- [ ] 数据：CounterFact 子集（可固定 1k/2k 条）+ 随机种子 {1, 2, 3}；
- [ ] 指标：ES/PS/NS/GE/S；时间/显存记录（粗略）。

预期产出：批处理脚本（Windows CMD/PowerShell 或 Bash），以及聚合结果脚本 `experiments/summarize.py` 更新输出图表（PNG/CSV）。

## 8. 评测与可视化

- [ ] 跑 `downstream_eval/*` 与 `glue_eval/*`（可选），对比不同方法与超参；
- [ ] 生成历史编辑命中率与遗忘曲线（按编辑序列步数作横轴）；
- [ ] 输出最终表格与图：方法 × 批量 × r × rank 的指标对比。

预期产出：`figs/*.png` 与 `results/final_summary.csv`。

## 9. 写作与整理

- [ ] 撰写方法与推导简述：EMMET 闭式解、Replay/PEFT 如何嵌入与权衡；
- [ ] 记录实验设定、硬件/显存、运行时间、失败案例；
- [ ] 结论与未来工作（更大的模型/更稳的数值解/更强的 Adapter 结构）。

## 10. 里程碑（以当前日期为准）

- D0～D2：完成环境与最小运行验证；熟悉代码路径；
- D3～D6：跑通三大基线在小子集；得到首版 `baselines.csv`；
- D7～D10：实现 Replay（首版），做小规模消融；
- D11～D14：接入 LoRA/Adapters（首版），与 Replay 组合再验证；
- D15～D18：扩大量级（至 1k+），完善脚本与可视化；
- D19～D21：汇总结果、撰写报告与清理仓库。

## 11. 风险与备选方案

- [ ] 显存不足：优先小模型/更小 batch，或启用 8-bit/4-bit 量化；分层逐次编辑；
- [ ] 数值不稳定：对闭式解加入正则/阻尼，或限制历史样本拼接规模；
- [ ] 速度瓶颈：预先缓存中间量（K/Z），减少重复计算；
- [ ] 复现偏差：固定随机种子、记录版本与超参，保留失败日志用于分析。

---

附：可选命令片段（Windows CMD，按需调整环境名/路径）

```plaintext
:: 创建并激活环境（优先 conda）
conda create -n emmet-replay python=3.10 -y
conda activate emmet-replay

:: 安装依赖（示例，具体以仓库为准）
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
pip install -r unified-model-editing/requirements.txt

:: 运行一个最小数据采样与评测（示例）
python unified-model-editing/create_samples_cf.py --n_samples 200
python unified-model-editing/emmet/emmet_main.py --hparams unified-model-editing/hparams/EMMET/some_config.json
python unified-model-editing/downstream_eval/current_edit_scores.py --results_dir results/baselines
```
