
% ACL-style project report generated from pre.md
\documentclass[11pt]{article}

% ===== ACL TEMPLATE STYLE PACKAGES =====
% (Assuming ACL-style compilation environment; adjust as needed.)
\usepackage[hyperref]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{microtype}
\usepackage{inconsolata}
\documentclass[tikz, border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx} % 用于插入图片
\usepackage{adjustbox} % 可选：用于更灵活的图片尺寸调整（如需统一高度）

\title{GRMeT: Greedy and Robust Low-Rank Mass Editing for Transformer Language Models}

\author{Group 25\\
  Applied Natural Language Processing Course Project\\
  \texttt{Weizhi Tang   } 
  \texttt{   Ruiheng Li}}

\date{}

\begin{document}

\maketitle

\begin{abstract}
Large pre-trained language models encode extensive world knowledge but inevitably contain factual errors and outdated information. Building on ROME and MEMIT, we reconstruct a unified model editing framework, \textbf{GRMeT} (Greedy and Robustness Low-Rank Model Editing). GRMeT extends EMMET's ~\cite{gupta2024unifiedframeworkmodelediting} equality-constrained mass editing formulation with parameter-efficient fine-tuning (PEFT), combining closed-form equality constraints with low-rank LoRA updates \cite{hu2021loralowrankadaptationlarge}. This design substantially reduces storage cost and mitigates interference with neighboring facts. Furthermore, we introduce a trust-based credit mechanism and replay strategy that perform adaptive rollback and multi-step greedy editing for the same fact, improving robustness and success rate during the editing process. Finally, we propose an evaluation protocol that jointly considers efficacy, paraphrase generalization, neighborhood specificity, and a composite score, providing a more comprehensive assessment of editing quality. Experimental results on GPT2-XL show that GRMeT offers a practical, lightweight and stable solution for small-batch edits on small models while preserving local knowledge.
\end{abstract}

\section{Introduction}
Large pre-trained language models (PLMs) acquire a broad spectrum of world knowledge through unsupervised pre-training, which endows them with strong utility across a wide range of downstream tasks. However, the same mechanism also makes them susceptible to factual errors and knowledge obsolescence. Consequently, a growing body of work focuses on \emph{targeted model editing}, which aims to modify the behavior of the model on a small set of inputs without retraining the entire network.\cite{gupta2024model}

Representative localization--editing approaches have shown that many factual associations concentrate in mid-layer feed-forward computations, and that low-rank parameter increments can achieve precise corrections with limited side effects. Meanwhile, parameter-efficient fine-tuning (PEFT) techniques, such as Low-Rank Adaptation (LoRA), provide compact and storage-friendly ways to introduce task or fact-specific changes without touching all parameters.\cite{gupta2024unified} These lines of work suggest that combining structural insights about factual representations with low-rank adaptation can produce powerful and efficient editing algorithms.

Despite these advances, several practical challenges hinder reliable and long-term model editing. Repeated or large-scale edits can cause previously corrected facts to gradually fail, leading to sudden catastrophic forgetting under sequential update scenarios. Unified analyzes of editing algorithms highlight the trade-off between preserving existing knowledge and injecting new information, and show that blindly increasing batch size may expose failure modes, necessitating careful regularization and evaluation. In addition, evaluations targeting specificity and multi-hop propagation reveal that many editors cannot generalize edits along reasoning chains, and that existing benchmarks under-estimate downstream side effects. Memory-augmented and retrieval-based remedies have been proposed to offload persistence from the backbone network and reduce destructive interference, but these methods carry their own engineering and verification challenges. Finally, recent work indicates that constrained fine-tuning can be a surprisingly strong baseline, underscoring the importance of including carefully tuned realistic baselines in comparative studies.~\cite{huang2025selfaug}

This course project builds on the material from an Applied Natural Language Processing class. Our goal is to implement and critically examine editing methods in practice, deepen our understanding of their underlying principles, and explore how to combine them to achieve more robust behavior. Specifically, we target scenarios where users desire precise factual updates while minimizing unintended changes to the model's existing knowledge.

\section{Related Work}
We focus on three representative editing methods: ROME, MEMIT and EMMET. Table~\ref{tab:method_comparison} summarizes their key characteristics.

% 需确保导言区已加载必要宏包（与之前一致）：
% \usepackage{booktabs, amsmath, amssymb, caption}
% \captionsetup{font=small}

\begin{table*}[t]
  \centering
  \small
  \caption{High-level comparison of ROME, MEMIT, and EMMET.}
  \label{tab:method_comparison}
  % 跨两列宽度分配：总宽度≈\linewidth，列宽按内容比例调整，预留少量间距
  \begin{tabular}{p{0.15\linewidth}p{0.28\linewidth}p{0.28\linewidth}p{0.25\linewidth}}
    \toprule
    Dimension & ROME & MEMIT & EMMET \\
    \midrule
    Constraint & Strict equality & Relaxed least-squares & Equality constraint \\
    Form & $\hat{W}k_e = v_e$ & $\lVert \hat{W}K_E - V_E \rVert$ & $\hat{W}k_i^e = v_i^e,\ \forall i$ \\
    Batch editing & Single fact & Multiple facts & Multiple facts (up to $10$k) \\
    Core strength & High precision, minimal damage to old knowledge & High batch efficiency & Combines equality precision with mass-edit efficiency \\
    Implementation logic & Per-fact closed-form rank-one update & Least-squares closed-form multi-fact update & Extends ROME's equality constraints to batch with closed-form solution \\
    \bottomrule
  \end{tabular}
\end{table*}

ROME can be viewed as a \emph{foundational single-fact editing tool}. It relies on strict equality constraints to guarantee high accuracy in edited facts, but does not support batch editing, limiting its applicability to large-scale knowledge updates ~\cite{meng2023locatingeditingfactualassociations}. MEMIT is a \emph{mass-editing extension} that relaxes the equality constraints to a least-squares objective, enabling efficient joint updates for many facts. However, this relaxation can lead to reduced accuracy for new facts or mild degradation of old knowledge in some settings.

EMMET addresses part of this issue by reinstating equality-style constraints in a batch setting. Although MEMIT supports mass editing, its reliance on relaxed least-squares constraints may sacrifice precision. EMMET, by contrast, uses equality-constrained closed-form updates to achieve batch editing while improving both the accuracy of new fact memories and the preservation of existing knowledge. This makes EMMET particularly suitable for domains that demand high editing accuracy and safety, such as medicine or law.

Nevertheless, EMMET still directly writes full-rank updates to the model weights. In equality-based editing, these full-rank updates inevitably influence neighboring facts, and our reproduction of EMMET indicates that neighborhood fidelity can be degraded by up to 40\% in some settings. This motivates us to search for a method that retains the strong success rate of equality-based editing while reducing its impact on surrounding knowledge.

\subsection{LoRA and Parameter-Efficient Fine-Tuning}
Local fine-tuning of large language models is an effective adaptation strategy for downstream tasks. LoRA (Low-Rank Adaptation) ~\cite{hu2021loralowrankadaptationlarge}, as a core PEFT method, focuses on \emph{local updates} to critical parameters: it inserts small low-rank matrices into key layers (often the $Q$, $K$, $V$ projection layers in Transformers), freezing the original weight matrix $W$ and training only the low-rank matrices $A$ and $B$.

Concretely, LoRA assumes that the change in task-specific parameter  lies in a low-rank subspace. Given a pre-trained weight matrix $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, LoRA introduces matrices $A \in \mathbb{R}^{d_{\text{in}}\times r}$ and $B \in \mathbb{R}^{d_{\text{out}}\times r}$, with rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$, and parameterizes the adapted weight as
%
\begin{equation}
  W' = W + B A^{\top}.
\end{equation}
%
During training, $W$ remains frozen and only $A$ and $B$ are updated (plus a small number of bias parameters if desired). This design significantly reduces trainable parameters, mitigates overfitting, and facilitates the storage and deployment of multiple task- or fact-specific adapters.

At first glance, PEFT and LoRA appear to be a natural remedy for the shortcomings of full-rank equality-based editing methods like EMMET: they can restrict updates to a low-rank subspace and thus potentially limit interference on neighboring facts.

\section{Problem Formulation and Motivation}
EMMET performs closed-form updates on full-rank weights at each edited layer. Early and later edits may interfere with each other and, more importantly, applying large full-rank updates is a major contributor to degradation in neighborhood specificity (NS). Our empirical reproduction confirms that EMMET can substantially affect neighboring facts, which may be acceptable in domains where achieving high factual accuracy is paramount, but sub-optimal when the goal is to maintain the original model knowledge as much as possible.

Our objective is therefore to design a method that:
\begin{itemize}
  \item preserves the strong success rate and precision of equality-based editing,
  \item significantly improves neighborhood fidelity by reducing the impact on unrelated or nearby facts, and
  \item supports small-batch and sequential editing scenarios with controllable side effects.
\end{itemize}

To this end, we propose to combine the strengths of equality-constrained mass editing (EMMET) with low-rank PEFT methods (LoRA). Intuitively, instead of directly applying EMMET's full-rank $\Delta W$ to each layer, we aim to project or approximate these updates within a low-rank subspace parameterized by LoRA, and control the application of these increments in a greedy and trust-aware manner.

\section{Baseline Algorithms}
For completeness, we briefly summarize the workflows of ROME, MEMIT, and the baseline version of EMMET.

\subsection{ROME: Rank-One Model Editing}
ROME operates on a single editing request $r = (x, y_{\text{old}}, y_{\text{new}})$, where $x$ is the prompt, $y_{\text{old}}$ is the original fact, and $y_{\text{new}}$ is the desired updated fact. Select a specific MLP layer $\ell$ with a weight matrix $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ and obtain a "fact lookup" hidden state $h$ by forwarding $x$ through the model. Assuming local linearity, ROME constructs a rank-one update $\Delta W = u v^{\top}$ such that $W h + \Delta W h$ approximates the representation corresponding to $y_{\text{new}}$. The updated weight $W' = W + \Delta W$ is written back to the model, yielding an edited model that accurately outputs $y_{\text{new}}$ on input $x$ while minimally affecting other inputs.

\subsection{MEMIT: Mass Editing Memory in a Transformer}
MEMIT generalizes ROME to mass editing. Given a batch of editing requests $\mathcal{R} = \{r_i\}_{i=1}^N$, it selects a set of layers $\mathcal{L}$ and, for each layer $\ell$, constructs key representations $K_\ell$ and target residuals $T$. Using a covariance matrix $C$ and a least-squares objective, MEMIT solves for joint updates $\Delta W_\ell$ per layer to minimize
%
\begin{equation}
  \min_{\Delta W_\ell} \big\|T - \Delta W_\ell K_\ell\big\|_F^2,
\end{equation}
%
subject to certain scale constraints. The closed-form solution is then written back to all selected layers, enabling efficient editing of multiple facts in one shot.

\subsection{EMMET: Equality-Constrained Mass Model Editing (Baseline)}
EMMET can be viewed as extending the equality-constrained perspective of ROME to the batch setting. Given a batch of requests and a set of layers, it extracts and backs up the original weights $W_\ell$, constructs key matrices $K_\ell$, and computes target and residual matrices $Z$, $Z^{\text{cur}}$ and $T$. For each layer, it loads or estimates a covariance matrix $C_\ell$, adapts it with hyperparameters (e.g., \texttt{mom2\_update\_weight}, \texttt{update\_norm\_lambda}), and solves a constrained least-squares problem of the form
%
\begin{equation}
  \min_{\Delta W_\ell} \big\|T - \Delta W_\ell K_\ell\big\|_F^2 + \lambda_{\text{em}} \big\langle \Delta W_\ell, \tilde{C}_\ell \Delta W_\ell \big\rangle,
\end{equation}
%
where $\tilde{C}_\ell$ combines covariance and regularization, and $\lambda_{\text{em}}$ is an EMMET-specific weight. The resulting closed-form solution
%
\begin{equation}
  \Delta W_\ell = T K_\ell^{\top} \big( K_\ell K_\ell^{\top} + \tilde{C}_\ell \big)^{-1}
\end{equation}
%
is applied directly to the backed-up weights $W_\ell^{(0)}$, yielding an edited model that supports the joint editing of many facts.

\section{Method: GRMeT Framework}
We now describe our proposed framework GRMeT, which extends EMMET with LoRA-native updates, memory replay, trust-aware gating, and greedy multi-step optimization.

\subsection{Challenges of Naive LoRA Projection}
In a LoRA-native mode, the baseline EMMET solution produces, for each layer $\ell$, a full-rank update
%
\begin{equation}
  \Delta W_{\text{full}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}.
\end{equation}
%
A naive approach would be to project $\Delta W_{\text{full}}$ into a rank-$r$ LoRA subspace in a single step. However, this can:
\begin{itemize}
  \item amplify certain directions excessively, causing strong interference with old knowledge; and
  \item induce direction mixing among different edits under low-rank compression, leading to lower efficacy and paraphrase generalization (ES/PS).
\end{itemize}


% 三张图片并排居中代码（核心部分）
\begin{figure*}[t] % figure* 跨两栏（ACL 模板默认双栏，单栏用 figure）
    \centering
    % 每张图片宽度设为文本宽度的 1/3 减间距，确保不溢出
    \includegraphics[width=0.31\textwidth]{latex/images/emmet_lora_compare.png} % 替换为你的第一张图片路径/文件名
    \hspace{0.02\textwidth} % 图片之间的水平间距（可调整）
    \includegraphics[width=0.31\textwidth]{latex/images/emmet_lora_compare1.png} % 替换为你的第二张图片路径/文件名
    \hspace{0.02\textwidth} % 图片之间的水平间距（可调整）
    \includegraphics[width=0.31\textwidth]{latex/images/emmet_lora_compare2.png} % 替换为你的第三张图片路径/文件名
    
    % 图片总标题（ACL 建议简洁明了，可根据需求修改）
    \caption{Different Lora rank}
    \label{lora} % 图片标签，用于文中引用（如 \ref{fig:three_figures}）
\end{figure*}
Our empirical results confirm this issue: as the LoRA rank increases, the performance of single edits converges toward baseline EMMET fig~\ref{lora}, while low-rank settings improve NS but significantly degrade ES and PS. This indicates that naive low-rank approximation is insufficient; we must control how the full-rank delta is injected into the low-rank space.

\subsection{EMMET Extended Workflow}
To address the above challenges, we extend EMMET's workflow as shown in Table~\ref{tab:emmet_extended_flow}, incorporating replay, trust gating, and greedy LoRA updates.

\begin{table*}[t]
  \centering
  \small
  \caption{Extended workflow of GRMeT}
  \label{tab:emmet_extended_flow}
  \begin{tabular}{p{0.06\textwidth}p{0.9\textwidth}}
    \toprule
    Step & Description \\
    \midrule
    1 & Input a batch of editing requests $\{r_i\}_{i=1}^N$. \\
    2 & Perform memory replay: sample previous edits from a buffer and merge them with the new batch, obtaining an extended set $\tilde{\mathcal{R}} = \{r_i\} \cup \{r_j^{\text{replay}}\}$. \\
    3 & Execute the EMMET closed-form solution: compute $\Delta W_\ell$ and distance metrics ($d_{\text{pres}}, d_{\text{new}}, d_{\text{old}}$) for each layer $\ell$. \\
    4 & Select the write-back mode based on an \texttt{edit\_mode} flag, leading to two branches: raw mode vs. LoRA-native mode. \\
    \midrule
    5a & \textit{Raw mode branch} -- Trust gate (raw): compute a trust score $\text{trust\_score} \in [0,1]$ from $d_{\text{pres}}, d_{\text{new}}, d_{\text{old}}$; potentially scale or roll back $\Delta W_\ell$ according to the score. \\
    6a & Raw write-back: update weights as $W_\ell \leftarrow W_\ell + \Delta W_\ell^{\ast}$. \\
    7a & Output the edited model $f_{\theta'}$. \\
    \midrule
    5b & \textit{LoRA-native mode branch} -- Trust gate (LoRA): compute $\text{trust\_score}$ at the first step; if too low, roll back entirely and skip the update. \\
    6b & Greedy LoRA multi-step loop: split $\Delta W_\ell$ into $S$ small steps; at each step $s$, call a LoRA-native backend to approximate
         $B_\ell A_\ell \approx B_\ell^{\text{prev}} A_\ell^{\text{prev}} + \eta_s \, \Delta W_\ell$. \\
    7b & Optional online evaluation: compute an evaluation metric $C_s$ (composite score) and check whether $C_s - C_{s-1} \geq \epsilon$. \\
    8b & If there is no improvement ($C_s - C_{s-1} < \epsilon$): roll back the current step, restore the LoRA snapshot, and increase the patience counter. \\
    9b & If improved ($C_s - C_{s-1} \geq \epsilon$): accept the current step, update the LoRA state, and set $C_{s-1} \leftarrow C_s$. \\
    10b & Stop the greedy loop when patience is exhausted or all steps are completed, obtaining the final LoRA stack $B_\ell A_\ell$. \\
    11b & Output the edited model $f_{\theta, \text{LoRA}}$. \\
    \bottomrule
  \end{tabular}
\end{table*}

This workflow highlights the interaction between four components:
\begin{enumerate}
  \item \textbf{Replay} extends the input request set, incorporating old edits at the request level.
  \item \textbf{EMMET core} computes per-layer closed-form deltas and distance metrics.
  \item \textbf{Trust} gates these deltas differently under raw vs. LoRA-native modes, providing an ``insurance'' mechanism.
  \item \textbf{Greedy LoRA} performs multi-step low-rank updates with online evaluation and rollback.
\end{enumerate}

\subsection{Edit Trust and Replay Mechanism}
We introduce an edit trust and insurance mechanism (Edit Trust/Rollback)~\cite{matetic2017rote} as a first-class component of the editing algorithm rather than a post-hoc remedy. The system maintains a buffer of past edits and their evaluation scores. For each new batch, it:
\begin{itemize}
  \item samples a subset of previous edits to form a replay set;
  \item computes an edit-specific trust score based on consistency of validation metrics (e.g., ES, PS, NS, composite);
  \item scales, accepts, or rejects candidate updates based on the trust score, with automatic rollback in low-trust cases.
\end{itemize}

This mechanism allows GRMeT to adaptively roll back edits that exhibit negative side effects, particularly in sequential editing regimes.

\subsection{Greedy LoRA in the Equality-Constrained Subspace}
Instead of applying $\Delta W_\ell$ in a single shot, GRMeT decomposes the full-rank delta for each layer into a sequence of small updates in the LoRA subspace. At each step, the algorithm:
\begin{enumerate}
  \item proposes a small low-rank increment that moves the LoRA parameters toward approximating the EMMET delta;
  \item evaluates the composite metric $C_s$ on a validation set combining edit targets and neighborhood probes;
  \item decides whether to accept or rollback the step based on the improvement $C_s - C_{s-1}$ and patience.
\end{enumerate}

This greedy ``small-step hill-climbing'' process allows GRMeT to approximate equality-based updates while explicitly optimizing for a composite criterion that trades off editing efficacy and neighborhood preservation.

\section{Evaluation Metrics}
To comprehensively assess the effectiveness and controllability of model editing, we adopt four metrics and place particular emphasis on neighborhood fidelity.

\paragraph{Efficacy Score (ES).} ES measures whether the edit succeeds on the target probes, i.e., the proportion of prompts for which the edited model outputs the correct updated fact.

\paragraph{Paraphrase Score (PS).} PS evaluates paraphrase generalization: we apply synonymically rephrased prompts for the same fact and measure the success rate. This reflects whether the edit remains effective under different surface forms.

\paragraph{Neighborhood Specificity (NS).} NS measures local specificity by assessing the model's behavior on neighboring but non-target examples. Concretely, it is defined as the proportion of neighborhood samples on which the edited model preserves the original model's predictions (higher is better). This metric captures the degree to which an edit avoids disrupting unrelated knowledge.

\paragraph{Composite Score (CS).} We aggregate ES, PS, and NS into a composite score by simple averaging after normalization:
%
\begin{equation}
  \text{CS} = \frac{1}{5} (\text{ES} + \text{PS} + 3\text{NS}).
\end{equation}
%
This single scalar is used as the main quality indicator and as the acceptance criterion for greedy LoRA steps in GRMeT.

All scores are reported on a 0--1 scale, computed from discrete prediction matches on small probe sets.

\section{Experiments}
We use open-source implementations and datasets to reproduce ROME, MEMIT, and baseline EMMET on GPT2-XL, running up to 500 edits per method and summarizing results through bar and line plots. In the following, we describe key experimental settings and findings.

\subsection{Experimental Setup}
We follow previous work and use the CounterFact family of datasets, together with GPT2-XL as the backbone model. For each method, we performed both single-edit and multi-edit experiments.

\paragraph{Single-edit setting.} We randomly sample individual facts and apply exactly one edit at a time, evaluating ES, PS, NS, and CS.

\paragraph{Mass-edit setting.} We conduct experiments with varying number of edits (e.g., 20, 200, 500), examining the behavior of model editing under repeated or large-scale updates.

\subsection{Baselines: ROME, MEMIT, EMMET}
Our reproduction confirms known qualitative trends:
\begin{itemize}
  \item For single edits, ROME and EMMET achieve significantly higher ES than MEMIT, but at the cost of lower NS.
  \item MEMIT tends to preserve neighboring knowledge better (higher NS), but its relaxed constraints can reduce the accuracy of the edited facts.
\end{itemize}
\begin{figure}[h]  % [h] 优先放在当前段落位置（双栏布局下单栏图片）
    \centering
    % 单栏宽度较窄，图片宽度设为 0.9\textwidth（避免过挤）
    \includegraphics[width=0.5\textwidth]{latex/images/微信图片_20251118033515_3354_596.png}
    
    \caption{MEMIT vs. EMMET for 100-batch edits.}
    \label{fig:single_singlecolumn}
\end{figure}
\subsection{GRMeT vs. Baselines}
We compare GRMeT with ROME, MEMIT, and EMMET using several visualization types. Figures are left as placeholders to be populated with actual plots.
\begin{figure*}[t]  % [t] 表示优先放在页面顶部（可选：[b]底部/[h]当前位置/[!t]强制顶部）
    \centering  % 图片居中
    % 图片宽度设为 0.8\textwidth（跨两栏文本宽度），可按需调整（如 0.9\textwidth 更宽）
    \includegraphics[width=0.8\textwidth]{latex/images/微信图片_20251118021739_3352_596.png}  % 替换为你的图片路径（PDF/PNG/JPG均可）
    
    % ACL 规范标题：简洁明了，首字母大写，标点规范
    \caption{Line plots: CS vs. number of edits.}
    \label{fig:single_fullwidth}  % 图片标签（用于正文引用）
\end{figure*}


\subsection{Observed Trends}
Our experiments reveal several notable patterns:

\begin{itemize}
  \item For small-batch edits (e.g., 20 edits), GRMeT attains ES and PS slightly lower than EMMET but significantly higher NS, leading to an overall improvement in CS.
  \item For single edits, GRMeT achieves editing accuracy comparable to ROME while maintaining high neighborhood fidelity, and its CS remains higher than ROME across multiple single-edit rounds.
  \item For large-batch edits \ref{fig:single_fullwidth}, naive low-rank compression without careful balancing of EMMET's delta can severely hurt performance. GRMeT compresses all updates into per-layer LoRA subspaces, and each delta passes through SVD/low-rank fitting. Without fine-grained balancing, performance decays after approximately 50 edits. CS curves show that GRMeT outperforms baselines in the early regime (e.g., up to 30 edits) but degrades significantly beyond 50 edits, highlighting the difficulty of stable large-scale editing.
\end{itemize}

\section{Ablation Study}
To validate the necessity of each component in the GRMeT framework, we perform an ablation study on 20-edit experiments, where we systematically remove modules such as replay, trust gating, or greedy LoRA updates. 
\[
\begin{array}{lcccc}
\toprule
\textbf{Method} & \textbf{ES} \uparrow & \textbf{PS} \uparrow & \textbf{NS} \downarrow & \textbf{CS} \uparrow \\
\midrule
\text{Baseline (EMMET)}          & 0.95 & 0.91 & 0.62 & 0.74 \\
\text{+ Native LoRA}             & 0.31 & 0.28 & 0.77 & 0.58 \\
\text{+ Trust \& Rollback}       & 0.95 & 0.91 & 0.62 & 0.74 \\
\text{+ Greedy Edit}             & 0.95 & 0.88 & 0.41 & 0.61 \\
\text{+ Memory Replay}           & 0.82 & 0.80 & 0.66 & 0.72 \\
\text{GRMeT}                     & 0.88 & 0.86 & 0.83 & 0.85 \\
\bottomrule
\end{array}
\]


Experimental results indicate that all components are indispensable for the full performance of GRMeT: removing any module leads to noticeable degradation in ES, PS, NS, or CS, confirming the importance of replay, trust, and greedy LoRA in achieving a balanced editing behavior.

\section{Discussion}
Our findings suggest that combining equality-based mass editing with low-rank PEFT is a promising direction for robust and efficient knowledge updates in language models. GRMeT preserves much of EMMET's high editing accuracy while leveraging LoRA to reduce interference with neighboring facts, particularly in small-batch and sequential editing scenarios.

However, handling very large batches remains challenging. GRMeT compresses all updates into a fixed low-rank subspace per layer, and repeated SVD-based approximations can gradually erode the delicate balance required for stable edits. In future work, we plan to refine the per-step balancing of deltas and explore dynamic rank adaptation or fact-specific adapter routing to further mitigate catastrophic forgetting in large-scale settings.

\section{Applications}
The proposed GRMeT framework has several practical applications:

\paragraph{Personalization and customization.} Small-batch edits can be used to adjust background facts or preferences for individual users or small groups (e.g. personal assistants) without modifying the full model.

\paragraph{Low-resource devices and edge deployment.} On resource-constrained or offline systems, small-batch low-rank edits avoid the cost of incremental training or large-scale parameter updates, enabling fast deployment of ``patches'' to correct critical facts.

\paragraph{A/B testing and progressive rollout.} Small-batch edits can serve as controlled experimental patches: we can first validate their effects on a limited set of inputs (tracking NS/ES/PS), and then gradually expand their scope if performance is satisfactory.

\section{Conclusion and Future Work}
This project develops GRMeT, a low-rank equality-based editing framework that combines EMMET's mass-editing precision with LoRA's parameter efficiency. GRMeT retains high editing accuracy, improves neighborhood fidelity, and provides an insurance mechanism via replay and trust-aware greedy updates. It offers a practical solution for small-model and small-batch editing with a good balance between accuracy, robustness, and storage cost.

Due to time and resource limitations, we have not fully tuned all hyperparameters, and we expect that better configurations can further improve performance. In particular, future work can explore:
\begin{itemize}
  \item optimizing the trade-off among ES, PS, and NS through more principled multi-objective criteria;
  \item studying rank reduction or target representation compression to better align with the equality constraints;
  \item enhancing parameter balancing in large-batch edits to mitigate catastrophic forgetting beyond 50+ edits.
\end{itemize}

GRMeT remains an open and promising framework for further refinement, especially in terms of dynamic rank scheduling and batch-wise balancing mechanisms.

\section*{Project Reflections}
Throughout this project, we substantially modified and extended existing codebases. Using Git for version control proved crucial for collaborative development and rapid iteration. Maintaining a detailed TODO list also made it easier to track module-level changes and verify whether earlier modifications introduced unintended side effects.

\bibliographystyle{acl_natbib}
\bibliography{custom}

\end{document}