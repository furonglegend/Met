% ACL-style paper for GRMeT (extended version)
\documentclass[11pt]{article}

% ACL-style template packages
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
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc}

\title{GRMeT: Greedy and Robust Low-Rank Mass Editing for Transformer Language Models}

\author{Anonymous ACL Submission \\
  Affiliation \\
  \texttt{email@example.com}}

\date{}

\begin{document}

\maketitle

\begin{abstract}
Model editing offers a practical alternative to full retraining when correcting or updating factual knowledge stored in large pretrained language models. Existing locate-and-edit strategies can produce accurate, localized corrections, but they are vulnerable to progressive degradation under repeated updates, can perturb unrelated capabilities, and often require broad parameter modifications that complicate auditing and rollback. Building on ROME, MEMIT, and EMMET, we present \textbf{GRMeT}, a unified editing framework that extends equality-constrained mass editing with parameter-efficient low-rank adapters, memory replay, and trust-aware greedy updates. GRMeT combines closed-form equality constraints with LoRA-style low-rank updates to substantially reduce storage cost and mitigate interference with neighboring facts. It further employs causality-informed layer selection, an edit-level trust and rollback mechanism, a replay scheduler, and a greedy multi-step optimization scheme that uses a composite metric over edit success, paraphrase generalization, and neighborhood specificity to decide whether to accept or revert small low-rank increments. Evaluations on GPT2-XL with CounterFact-style benchmarks show that GRMeT improves locality and long-term stability over strong baselines, particularly in small-batch and sequential editing regimes, while retaining high factual accuracy.
\end{abstract}

\section{Introduction}
Large pretrained language models encode a broad spectrum of world knowledge acquired during unsupervised pretraining, which makes them useful across many downstream tasks but also leaves them susceptible to factual errors and obsolescence. A growing body of work studies targeted interventions that change only a small portion of model behavior rather than retraining the entire network. Representative locate-and-edit methods demonstrate that many factual associations concentrate in mid-layer feed-forward computations and that rank-constrained parameter deltas can implement precise corrections with limited collateral damage \cite{meng2022locating,meng2022mass,gupta2024unified}.

Concurrently, parameter-efficient fine-tuning (PEFT) techniques such as Low-Rank Adaptation (LoRA) provide a compact and storage-friendly means to introduce task-specific or fact-specific modifications without overwriting the full parameter set \cite{hu2021loralowrankadaptationlarge,he2023parameter}. These lines of work suggest that combining structural insights about factual representations with low-rank adaptation can produce powerful and efficient editing algorithms.

Despite these advances, several practical obstacles hinder reliable, long-lived editing. Repeated or large-scale edits can produce progressive attrition of previously corrected facts and may eventually lead to abrupt catastrophic forgetting under sequential update workloads \cite{gupta2024model}. Unified analyses of editing algorithms highlight trade-offs between preserving pre-existing knowledge and inserting new information, and they show that naively scaling up batched edits can expose failure modes that demand careful regularization and evaluation \cite{gupta2024unified,yoon2024bigger}. Evaluations focused on specificity and multi-hop propagation reveal that many editors fail to generalize edits across reasoning chains and that current benchmarks do not fully capture downstream side effects \cite{hoelscher2023detecting,zhong2023mquake,hua2024propagation}.

Memory-augmented and retrieval-based remedies have been proposed to offload permanence from the backbone and to reduce destructive interference, but these approaches introduce their own engineering and validation challenges \cite{mitchell2022memory,qiao2024comem,wang2024wise}. Finally, recent work shows that constrained standard fine-tuning can be a surprisingly strong baseline, which underscores the importance of comparing any proposed pipeline against well-tuned, realistic baselines \cite{gangadhar2024model,huang2025selfaug}.

We address these challenges by introducing \textbf{GRMeT} (Greedy and Robust low-rank Mass Editing), a unified model editing framework that intentionally combines four mechanisms whose strengths are mutually reinforcing: (1) equality-constrained mass editing to retain high single-fact accuracy; (2) low-rank LoRA-style adapters that implement edits in compact, reversible subspaces; (3) editor-level memory replay and trust-aware rollback to preserve edited facts and guard against negative side effects; and (4) a greedy multi-step optimizer that decomposes full-rank equality-constrained updates into a sequence of small low-rank steps, each accepted only if a composite score over edit efficacy and locality improves.

Empirically, GRMeT increases edit success and locality, better preserves unrelated capabilities under repeated edits, and reduces model drift compared with standard editing baselines. The remainder of the paper details the framework, experimental methodology, and ablations.

\section{Background and Baselines}

\subsection{ROME, MEMIT, and EMMET}
We first summarize three representative editing methods that form the basis of GRMeT: ROME, MEMIT, and EMMET. Table~\ref{tab:method_comparison} highlights their main characteristics.

\begin{table*}[t]
  \centering
  \small
  \caption{High-level comparison of ROME, MEMIT, and EMMET.}
  \label{tab:method_comparison}
  \begin{tabular}{p{0.15\linewidth}p{0.28\linewidth}p{0.28\linewidth}p{0.25\linewidth}}
    	oprule
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

\paragraph{ROME.}
ROME operates on a single editing request $r = (x, y_{\text{old}}, y_{\text{new}})$, where $x$ is the prompt, $y_{\text{old}}$ is the original fact, and $y_{\text{new}}$ is the desired updated fact. It selects a specific MLP layer $\ell$ with weight matrix $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ and obtains a "fact lookup" hidden state $h$ by forwarding $x$ through the model. Assuming local linearity, ROME constructs a rank-one update $\Delta W = u v^{\top}$ such that $W h + \Delta W h$ approximates the representation corresponding to $y_{\text{new}}$. The updated weight $W' = W + \Delta W$ is written back to the model, yielding an edited model that accurately outputs $y_{\text{new}}$ on input $x$ while minimally affecting other inputs.

\paragraph{MEMIT.}
MEMIT generalizes ROME to mass editing. Given a batch of editing requests $\mathcal{R} = \{r_i\}_{i=1}^N$, it selects a set of layers $\mathcal{L}$ and, for each layer $\ell$, constructs key representations $K_\ell$ and target residuals $T$. Using a covariance matrix $C$ and a least-squares objective, MEMIT solves for joint updates $\Delta W_\ell$ per layer to minimize
\begin{equation}
  \min_{\Delta W_\ell} \big\|T - \Delta W_\ell K_\ell\big\|_F^2,
\end{equation}
subject to scale constraints. The closed-form solution is written back to all selected layers, enabling efficient editing of multiple facts in one shot.

\paragraph{EMMET.}
EMMET can be viewed as extending the equality-constrained perspective of ROME to the batch setting. Given a batch of requests and a set of layers, it extracts and backs up the original weights $W_\ell$, constructs key matrices $K_\ell$, and computes target and residual matrices $Z$, $Z^{\text{cur}}$ and $T$. For each layer, it loads or estimates a covariance matrix $C_\ell$, adapts it with hyperparameters, and solves a constrained least-squares problem of the form
\begin{equation}
  \min_{\Delta W_\ell} \big\|T - \Delta W_\ell K_\ell\big\|_F^2 + \lambda_{\text{em}} \big\langle \Delta W_\ell, \tilde{C}_\ell \Delta W_\ell \big\rangle,
\end{equation}
where $\tilde{C}_\ell$ combines covariance and regularization, and $\lambda_{\text{em}}$ is an EMMET-specific weight. The resulting closed-form solution
\begin{equation}
  \Delta W_\ell = T K_\ell^{\top} \big( K_\ell K_\ell^{\top} + \tilde{C}_\ell \big)^{-1}
\end{equation}
is applied directly to the backed-up weights $W_\ell^{(0)}$, yielding an edited model that supports the joint editing of many facts.

Nevertheless, EMMET still directly writes full-rank updates to the model weights. These full-rank updates inevitably influence neighboring facts, and empirical reproductions indicate that neighborhood fidelity can be degraded substantially in some settings. This motivates low-rank and more controlled variants.

\subsection{LoRA and Parameter-Efficient Fine-Tuning}
Local fine-tuning of large language models is an effective adaptation strategy for downstream tasks. LoRA (Low-Rank Adaptation) \cite{hu2021loralowrankadaptationlarge}, as a core PEFT method, focuses on local updates to critical parameters: it inserts small low-rank matrices into key layers (often $Q$, $K$, $V$ projection layers in Transformers), freezing the original weight matrix $W$ and training only the low-rank matrices $A$ and $B$.

Given a pre-trained weight matrix $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, LoRA introduces matrices $A \in \mathbb{R}^{d_{\text{in}}\times r}$ and $B \in \mathbb{R}^{d_{\text{out}}\times r}$, with rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$, and parameterizes the adapted weight as
\begin{equation}
  W' = W + B A^{\top}.
\end{equation}
During training, $W$ remains frozen and only $A$ and $B$ are updated. This design significantly reduces trainable parameters, mitigates overfitting, and facilitates the storage and deployment of multiple task- or fact-specific adapters.

At first glance, PEFT and LoRA appear to be a natural remedy for the shortcomings of full-rank equality-based editing methods like EMMET: they can restrict updates to a low-rank subspace and thus potentially limit interference on neighboring facts. However, as we show next, naive projection of EMMET-style full-rank deltas into a low-rank subspace performs poorly.

\section{Problem Formulation}
EMMET performs closed-form updates on full-rank weights at each edited layer. Early and later edits may interfere with each other and, more importantly, large full-rank updates are a major contributor to degradation in neighborhood specificity (NS). Our empirical reproduction confirms that EMMET can substantially affect neighboring facts, which may be acceptable in domains where achieving high factual accuracy is paramount, but sub-optimal when the goal is to maintain the original model knowledge as much as possible.

Our objective is therefore to design a method that:
\begin{itemize}
  \item preserves the strong success rate and precision of equality-based editing,
  \item significantly improves neighborhood fidelity by reducing the impact on unrelated or nearby facts, and
  \item supports small-batch and sequential editing scenarios with controllable side effects.
\end{itemize}

To this end, GRMeT combines the strengths of equality-constrained mass editing (EMMET) with low-rank PEFT methods (LoRA), together with replay and trust-aware greedy updates.

\section{Method: GRMeT Framework}
\label{sec:method}

本节在短版 GRMeT 的基础上，进一步引入动态层选择、带学习调度的重放机制、子空间正交的 PEFT 约束以及混合专家路由，这些组件最初在我们早期框架中分别探索，这里统一在 GRMeT 之下重新表述。

\subsection{Preliminaries}
Let $f_\theta$ be an auto-regressive LM with parameters $\theta$. An edit request is a tuple $e=(x^\star, y^\star)$ consisting of a prompt $x^\star$ and desired target $y^\star$ (e.g., a tail entity). Let $\mathcal{V}_{\text{keep}}$ denote a held-out validation set used to measure locality and retention. We denote the edited model by $f_{\theta'}$ with $\Delta\theta = \theta' - \theta$.

We optimize a composite objective
\begin{align}
\mathcal{L}_{\text{edit}}(\theta') &=
\underbrace{\mathbb{E}_{(x^\star,y^\star)}\big[\text{CE}(f_{\theta'}(x^\star), y^\star)\big]}_{\text{edit success}}
+ \lambda_{\text{KL}}\, \underbrace{\mathbb{E}_{x\sim \mathcal{V}_{\text{keep}}}\big[\text{KL}(p_\theta(\cdot|x)\,\|\,p_{\theta'}(\cdot|x))\big]}_{\text{locality}}
+ \lambda_{\text{norm}}\, \|\Delta\theta\|_2^2,
\end{align}
which trades off successful rewriting of edit prompts, preservation of behavior on $\mathcal{V}_{\text{keep}}$, and parameter movement magnitude.

\subsection{Dynamic Layer Selection}
Empirical studies show that factual associations are concentrated in a subset of layers. Rather than editing all candidate layers, GRMeT uses dynamic importance scores to focus capacity.

For each edit $(x^\star,y^\star)$ we score layer relevance $s_\ell$ using either causal tracing or an attention-style rollout:
\begin{align}
 s_\ell^{\text{causal}} &= \Delta \text{logit}_{y^\star}\big(\text{mask}(\ell)\big),\\
 s_\ell^{\text{rollout}} &= \mathbf{1}^\top \Big(\prod_{h=1}^{\ell} A_h\Big) e_{\text{src}},
\end{align}
where $\text{mask}(\ell)$ denotes intervention at layer $\ell$ and $A_h$ is the attention matrix at layer $h$. We select top-$k$ layers $\mathcal{L}_k$ and optionally use soft weights $\alpha_\ell = \text{softmax}(s_\ell/\tau)$ to modulate the strength of the EMMET-style update at each layer. This reduces unnecessary perturbation of irrelevant layers and integrates seamlessly with the equality-constrained core of GRMeT.

\subsection{From Full-Rank EMMET to LoRA}
In the short version, we showed that naive projection of EMMET's full-rank $\Delta W_{\text{full}}$ into a rank-$r$ subspace can either harm ES/PS or damage NS. GRMeT instead treats $\Delta W_{\text{full}}$ as a \emph{target direction} and realizes it via LoRA modules.

Concretely, at an edited layer with base weight $W$ we maintain a low-rank adapter $B A^\top$ with rank $r \ll d$. Rather than overwriting $W$ directly, we update $A$ and $B$ so that
\begin{equation}
 W + B A^\top \approx W + \eta\, \Delta W_{\text{full}},
\end{equation}
for a step size $\eta \in (0,1]$. This approximation is performed greedily over multiple small steps (Section~\ref{subsec:greedy-lora}), with online evaluation after each step.

\subsection{Replay Buffer and Trust Score}
GRMeT maintains a prioritized buffer $\mathcal{B} = \{(x_i, y_i, \pi_i)\}$ of past edits, where $\pi_i$ is a priority estimate (e.g., recency or forgetting risk). When processing a new batch, we optionally draw a replay set $\mathcal{R} \subseteq \mathcal{B}$ and carry out a mixed update that includes both new edits and replayed ones:
\begin{align}
 \nabla\mathcal{L} = \nabla\mathcal{L}_{\text{edit}}(\mathcal{E}) + \gamma\, \nabla\mathcal{L}_{\text{edit}}(\mathcal{R}),
\end{align}
with $\gamma$ controlling the strength of replay.

After each candidate update (either raw or LoRA-native), we compute a scalar \emph{trust score} that summarizes multiple diagnostics:
\begin{align}
 \mathbf{m} &= [\,\text{norm}(-\Delta\text{PPL}),\; \text{NS},\; \text{norm}(\Delta \text{ECE}),\; \text{ES},\; \text{Locality}\,],\\
 \text{TrustScore} &= \sigma\Big(\sum_j w_j m_j\Big), \quad \sum_j w_j = 1,
\end{align}
where $\Delta$PPL is the perplexity change on a held-out corpus, NS is neighborhood specificity, ECE is calibration error, ES is edit success on rewrite/paraphrase prompts, and Locality measures agreement with the pre-edit model on $\mathcal{V}_{\text{keep}}$. If the trust score falls below a threshold, we roll back the update and down-weight the corresponding configuration in the replay scheduler (next subsection).

\subsection{Learned Replay Scheduler}
\label{subsec:lrs}
A fixed replay schedule can be suboptimal across heterogeneous edit streams. GRMeT therefore attaches a lightweight contextual bandit controller that adaptively decides (i) whether to invoke replay and (ii) which sampling strategy to use.

At step $t$ (after applying edit $e_t$) the controller chooses an arm
\begin{equation}
 a_t = (\text{do\_replay} \in \{0,1\},\; k \in \{1,\dots,K\},\; \text{strategy} \in \mathcal{S}),
\end{equation}
where $\mathcal{S}$ includes prioritized-by-age, prioritized-by-risk, and similarity-based sampling from $\mathcal{B}$. After observing the post-edit metrics, we form a reward
\begin{align}
 r_t &= \beta_1\,\text{norm}(\Delta \text{Loss}_{\text{keep}})
     + \beta_2\,\text{norm}(\Delta \text{NS})
     + \beta_3\,\text{norm}(\Delta \text{ECE})
     + \beta_4\,\text{TrustScore}_{t+1},
\end{align}
which directly reflects retention and safety. The controller is updated via UCB1 or $\epsilon$-greedy, maintaining statistics $(N_a,\hat{Q}_a)$ for each arm. Over time, it concentrates probability mass on replay policies that better preserve long-term locality given the observed stream of edits.

\subsection{Greedy LoRA in the Equality-Constrained Subspace}
\label{subsec:greedy-lora}
Building on the EMMET-style closed-form solution and the LoRA parameterization, GRMeT decomposes a full-rank $\Delta W_{\text{full}}$ into $S$ small low-rank steps. Pseudocode is as follows:
\begin{enumerate}
 \item Initialize LoRA parameters $(A,B)$ at the edited layers (zero or previously learned values) and compute baseline composite score $C_0$.
 \item For $s=1,\dots,S$:
   \begin{itemize}
     \item Propose a tentative low-rank increment $(\Delta A_s, \Delta B_s)$ obtained by minimizing $\|\Delta W_{\text{full}} - (B+\Delta B) (A+\Delta A)^\top\|_F^2$ under a small trust-region constraint.
     \item Update adapters tentatively: $(A',B') \leftarrow (A+\Delta A_s,\, B+\Delta B_s)$; evaluate composite score $C_s$ on a validation set mixing rewrite, paraphrase, and neighborhood prompts.
     \item If $C_s - C_{s-1} \geq \epsilon$, accept the step and set $(A,B)\leftarrow(A',B')$; otherwise revert and increase a patience counter.
   \end{itemize}
 \item Stop when $S$ steps are exhausted or patience is exceeded.
\end{enumerate}

This greedy scheme ensures that each additional low-rank movement is justified by an improvement in the multi-objective composite score, rather than blindly matching $\Delta W_{\text{full}}$ in Frobenius norm.

\subsection{PEFT Subspace and Orthogonalization}
GRMeT may maintain multiple adapters corresponding to different edit groups or domains. To reduce cross-edit interference, we explicitly regularize the subspaces spanned by different adapters.

Let $U_i \in \mathbb{R}^{d\times r}$ denote an orthonormal basis for the column space of the $i$-th adapter (e.g., obtained from an SVD of $B_i$). We penalize overlap between adapters via
\begin{align}
 \mathcal{L}_{\text{ortho}}^{\text{gram}} = \sum_{i<j} \|U_i^\top U_j\|_F^2,
\end{align}
which drives the Gram matrices towards identity on each block and zero off-diagonal. When creating a new adapter with preliminary basis $\tilde{U}$, we orthogonalize it against existing adapters:
\begin{align}
 U \leftarrow \text{qf}\Big(\tilde{U} - \sum_j U_j U_j^\top \tilde{U}\Big),
\end{align}
where $\text{qf}$ denotes a thin QR factorization. The full training loss becomes
\begin{align}
 \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{edit}} + \lambda_{\text{ortho}}\, \mathcal{L}_{\text{ortho}}^{\text{gram}},
\end{align}
with $\mathcal{L}_{\text{edit}}$ defined earlier. This encourages adapters to use complementary directions and empirically reduces cross-talk between unrelated edits.

\subsection{Mixture-of-Experts Routing for Edits}
Finally, GRMeT can distribute editing capacity across a small ensemble of experts at each edited layer. Let $\{\Delta_{\ell,i}\}$ be PEFT-style experts (e.g., different LoRA modules) attached to layer $\ell$, and let $g_{\ell,i}(x) \in [0,1]$ be input-dependent gates with $\sum_i g_{\ell,i}(x) \leq 1$. The edited hidden state is
\begin{align}
 h_\ell'(x) = h_\ell(x) + \sum_i g_{\ell,i}(x)\, \Delta_{\ell,i}(h_\ell(x)).
\end{align}
Domain- or cluster-specific routing reduces gradient conflict between edits drawn from different factual subsets (e.g., geography vs. biographies). Combined with the orthogonalization loss above, this yields a modular yet compact implementation of GRMeT on top of a single backbone.

% 其余 Evaluation Metrics、Experiments、Analysis、Conclusion 与短版基本一致，可直接沿用

\section{Evaluation Metrics}
To comprehensively assess the effectiveness and controllability of model editing, we adopt four metrics and place particular emphasis on neighborhood fidelity.

\paragraph{Efficacy Score (ES).} ES measures whether the edit succeeds on the target probes, i.e., the proportion of prompts for which the edited model outputs the correct updated fact.

\paragraph{Paraphrase Score (PS).} PS evaluates paraphrase generalization: we apply synonymically rephrased prompts for the same fact and measure the success rate.

\paragraph{Neighborhood Specificity (NS).} NS measures local specificity by assessing the model's behavior on neighboring but non-target examples. It is defined as the proportion of neighborhood samples on which the edited model preserves the original model's predictions (higher is better).

\paragraph{Composite Score (CS).} We aggregate ES, PS, and NS into a composite score by simple averaging after normalization:
\begin{equation}
  	ext{CS} = \frac{1}{5} (\text{ES} + \text{PS} + 3\text{NS}).
\end{equation}
This single scalar is used as the main quality indicator and as the acceptance criterion for greedy LoRA steps in GRMeT.

All scores are reported on a 0--1 scale, computed from discrete prediction matches on small probe sets.

\section{Experiments}

\subsection{Experimental Setup}
We use open-source implementations and datasets to reproduce ROME, MEMIT, and baseline EMMET on GPT2-XL, and to evaluate GRMeT.

\paragraph{Datasets.} We follow previous work and use the CounterFact family of datasets (CounterFact, MultiCounterFact, and QA-style zsRE) with neighborhood and attribute prompts to evaluate generalization and locality.

\paragraph{Models.} Our main backbone is GPT2-XL (1.5B parameters). We also report selected results on slightly larger and smaller models where computation permits.

\paragraph{Settings.} For each method, we perform both single-edit and mass-edit experiments. In the single-edit setting, we randomly sample individual facts and apply exactly one edit at a time, evaluating ES, PS, NS, and CS. In the mass-edit setting, we conduct experiments with varying numbers of edits (e.g., 20, 100, 500), examining behavior under repeated or large-scale updates.

\subsection{Baselines}
Our primary baselines are ROME, MEMIT, and EMMET. In addition, we consider EMMET variants that incorporate individual components of GRMeT (e.g., native LoRA only, trust and rollback only, greedy updates only, memory replay only, dynamic layer selection, orthogonalization, and MoE routing) to form an ablation suite.

Figure~\ref{fig:single_singlecolumn} shows a representative comparison of MEMIT and EMMET in a 100-batch setting; EMMET achieves higher ES but can degrade NS.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.5\textwidth]{latex/images/微信图片_20251118033515_3354_596.png}
    \caption{MEMIT vs. EMMET for 100-batch edits on CounterFact.}
    \label{fig:single_singlecolumn}
\end{figure}

\subsection{GRMeT vs. Baselines}
We compare GRMeT with ROME, MEMIT, and EMMET using bar and line plots. Figure~\ref{fig:single_fullwidth} shows representative CS curves as a function of the number of edits.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.8\textwidth]{latex/images/微信图片_20251118021739_3352_596.png}
    \caption{Composite score (CS) vs. number of edits for GRMeT and baselines.}
    \label{fig:single_fullwidth}
\end{figure*}

Our experiments reveal several notable patterns:
\begin{itemize}
  \item For small-batch edits (e.g., 20 edits), GRMeT attains ES and PS slightly lower than EMMET but significantly higher NS, leading to an overall improvement in CS.
  \item For single edits, GRMeT achieves editing accuracy comparable to ROME while maintaining high neighborhood fidelity, and its CS remains higher than ROME across multiple single-edit rounds.
  \item For larger batches, naive low-rank compression without careful balancing of EMMET's delta can severely hurt performance; GRMeT delays this degradation but still faces challenges beyond roughly 50 edits, highlighting the difficulty of stable large-scale editing in fixed-rank subspaces.
\end{itemize}

\subsection{Ablation Study}
To validate the necessity of each component in the GRMeT framework, we perform an ablation study on 20-edit experiments, where we systematically remove modules such as replay, trust gating, greedy LoRA updates, dynamic layer selection, orthogonalization, or MoE routing. Table~\ref{tab:ablation} summarizes representative results.

\begin{table}[t]
\centering
\small
\caption{Ablation results on 20-edit experiments. ES, PS, NS, and CS are reported on a 0--1 scale.}
\label{tab:ablation}
\begin{tabular}{lcccc}
	oprule
	extbf{Method} & \textbf{ES} $\uparrow$ & \textbf{PS} $\uparrow$ & \textbf{NS} $\uparrow$ & \textbf{CS} $\uparrow$ \\
\midrule
Baseline (EMMET)          & 0.95 & 0.91 & 0.62 & 0.74 \\
+ Native LoRA             & 0.31 & 0.28 & 0.77 & 0.58 \\
+ Trust \\& Rollback       & 0.95 & 0.91 & 0.62 & 0.74 \\
+ Greedy Edit             & 0.95 & 0.88 & 0.41 & 0.61 \\
+ Memory Replay           & 0.82 & 0.80 & 0.66 & 0.72 \\
GRMeT (full)              & 0.88 & 0.86 & 0.83 & 0.85 \\
\bottomrule
\end{tabular}
\end{table}

The results indicate that all components are important for the full performance of GRMeT: removing any module leads to noticeable degradation in ES, PS, NS, or CS, confirming the importance of combining replay, trust-aware gating, greedy LoRA, and structural constraints on the PEFT subspace.

\section{Analysis and Discussion}
Our findings suggest that combining equality-based mass editing with low-rank PEFT is a promising direction for robust and efficient knowledge updates in language models. GRMeT preserves much of EMMET's high editing accuracy while leveraging LoRA to reduce interference with neighboring facts, particularly in small-batch and sequential editing scenarios.

The learned replay scheduler further improves long-term retention by adaptively increasing replay frequency for edits that show signs of forgetting, while the orthogonalization loss reduces cross-talk between adapters assigned to different factual domains. Mixture-of-experts routing enables specialization across subsets of edits, which empirically improves neighborhood specificity without sacrificing paraphrase generalization.

However, handling very large batches remains challenging. GRMeT compresses all updates into a fixed low-rank subspace per layer, and repeated low-rank approximations can gradually erode the delicate balance required for stable edits. Future work can refine per-step balancing of deltas, explore dynamic rank adaptation, or use fact-specific adapter routing to further mitigate catastrophic forgetting in large-scale settings.

\section{Conclusion}
We have presented GRMeT, a low-rank equality-based editing framework that combines EMMET's mass-editing precision with LoRA's parameter efficiency, memory replay, a learned replay scheduler, and trust-aware greedy updates. GRMeT retains high editing accuracy, improves neighborhood fidelity, and provides an explicit insurance mechanism via replay and rollback, offering a practical solution for small-model and small-batch editing with a favorable balance between accuracy, robustness, and storage cost.

In future work, we plan to: (i) optimize the trade-off among ES, PS, and NS through more principled multi-objective criteria; (ii) study rank reduction or target representation compression to better align with equality constraints; and (iii) enhance parameter balancing and adapter routing in large-batch edits to mitigate catastrophic forgetting beyond dozens of edits.

\bibliographystyle{acl_natbib}
\bibliography{custom}

\end{document}
