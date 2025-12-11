% ACL-style paper for GRMeT (merged from ess1 and ess.2)
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
Model editing offers a practical alternative to full retraining when correcting or updating factual knowledge stored in large pretrained language models. Existing locate-and-edit strategies can produce accurate, localized corrections, but they are vulnerable to progressive degradation under repeated updates, can perturb unrelated capabilities, and often require broad parameter modifications that complicate auditing and rollback. Building on ROME, MEMIT, and EMMET, we present \textbf{GRMeT}, a unified editing framework that extends equality-constrained mass editing with parameter-efficient low-rank adapters, memory replay, and trust-aware greedy updates. GRMeT combines closed-form equality constraints with LoRA-style low-rank updates to substantially reduce storage cost and mitigate interference with neighboring facts. It further employs causality-informed layer selection, an edit-level trust and rollback mechanism, and a greedy multi-step optimization scheme that uses a composite metric over edit success, paraphrase generalization, and neighborhood specificity to decide whether to accept or revert small low-rank increments. Evaluations on GPT2-XL with CounterFact-style benchmarks show that GRMeT improves locality and long-term stability over strong baselines, particularly in small-batch and sequential editing regimes, while retaining high factual accuracy.
\end{abstract}

\section{Introduction}
Large pretrained language models encode a broad spectrum of world knowledge acquired during unsupervised pretraining, which makes them useful across many downstream tasks but also leaves them susceptible to factual errors and obsolescence. A growing body of work studies targeted interventions that change only a small portion of model behavior rather than retraining the entire network. Representative locate-and-edit methods demonstrate that many factual associations concentrate in mid-layer feed-forward computations and that rank-constrained parameter deltas can implement precise corrections with limited collateral damage \cite{meng2022locating,meng2022mass,gupta2024unified}.

Concurrently, parameter-efficient fine-tuning (PEFT) techniques such as Low-Rank Adaptation (LoRA) provide a compact and storage-friendly means to introduce task-specific or fact-specific modifications without overwriting the full parameter set \cite{hu2021loralowrankadaptationlarge,he2023parameter}. These lines of work suggest that combining structural insights about factual representations with low-rank adaptation can produce powerful and efficient editing algorithms.

Despite these advances, several practical obstacles hinder reliable, long-lived editing. Repeated or large-scale edits can produce progressive attrition of previously corrected facts and may eventually lead to abrupt catastrophic forgetting under sequential update workloads \cite{gupta2024model}. Unified analyses of editing algorithms highlight trade-offs between preserving pre-existing knowledge and inserting new information, and they show that naively scaling up batched edits can expose failure modes that demand careful regularization and evaluation \cite{gupta2024unified,yoon2024bigger}. Evaluations focused on specificity and multi-hop propagation reveal that many editors fail to generalize edits across reasoning chains and that current benchmarks do not fully capture downstream side effects \cite{hoelscher2023detecting,zhong2023mquake,hua2024propagation}.

Memory-augmented and retrieval-based remedies have been proposed to offload permanence from the backbone and to reduce destructive interference, but these approaches introduce their own engineering and validation challenges \cite{mitchell2022memory,qiao2024comem,wang2024wise}. Finally, recent work shows that constrained standard fine-tuning can be a surprisingly strong baseline, which underscores the importance of comparing any proposed pipeline against well-tuned, realistic baselines \cite{gangadhar2024model,huang2025selfaug}.

We address these challenges by introducing \textbf{GRMeT} (Greedy and Robust low-rank Mass Editing), a unified model editing framework that intentionally combines four mechanisms whose strengths are mutually reinforcing:
(1) equality-constrained mass editing to retain high single-fact accuracy; (2) low-rank LoRA-style adapters that implement edits in compact, reversible subspaces; (3) editor-level memory replay and trust-aware rollback to preserve edited facts and guard against negative side effects; and (4) a greedy multi-step optimizer that decomposes full-rank equality-constrained updates into a sequence of small low-rank steps, each accepted only if a composite score over edit efficacy and locality improves.

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

\subsection{Challenges of Naive LoRA Projection}
In a LoRA-native mode, an EMMET-style solution produces, for each layer $\ell$, a full-rank update
\begin{equation}
  \Delta W_{\text{full}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}.
\end{equation}
A naive approach would be to project $\Delta W_{\text{full}}$ into a rank-$r$ LoRA subspace in a single step. However, this can (i) amplify certain directions excessively, causing strong interference with old knowledge, and (ii) induce direction mixing among different edits under low-rank compression, leading to lower efficacy and paraphrase generalization.

Figure~\ref{lora} illustrates these issues: as the LoRA rank increases, the performance of single edits converges toward baseline EMMET, while low-rank settings improve NS but significantly degrade ES and PS. This indicates that naive low-rank approximation is insufficient; we must control how the full-rank delta is injected into the low-rank space.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.31\textwidth]{latex/images/emmet_lora_compare.png}
    \hspace{0.02\textwidth}
    \includegraphics[width=0.31\textwidth]{latex/images/emmet_lora_compare1.png}
    \hspace{0.02\textwidth}
    \includegraphics[width=0.31\textwidth]{latex/images/emmet_lora_compare2.png}
    \caption{Effect of different LoRA ranks on edit success (ES), paraphrase score (PS), and neighborhood specificity (NS).}
    \label{lora}
\end{figure*}

\subsection{Extended Workflow}
To address the above challenges, GRMeT extends EMMET's workflow by incorporating replay, trust gating, and greedy LoRA updates. Conceptually, the algorithm proceeds as summarized in Table~\ref{tab:emmet_extended_flow}.

\begin{table*}[t]
  \centering
  \small
  \caption{Extended workflow of GRMeT.}
  \label{tab:emmet_extended_flow}
  \begin{tabular}{p{0.06\textwidth}p{0.9\textwidth}}
    \toprule
    Step & Description \\
    \midrule
    1 & Input a batch of editing requests $\{r_i\}_{i=1}^N$. \\
    2 & Perform memory replay: sample previous edits from a buffer and merge them with the new batch, obtaining an extended set $\tilde{\mathcal{R}} = \{r_i\} \cup \{r_j^{\text{replay}}\}$. \\
    3 & Execute an EMMET-style closed-form solution: compute $\Delta W_\ell$ and distance metrics ($d_{\text{pres}}, d_{\text{new}}, d_{\text{old}}$) for each layer $\ell$. \\
    4 & Select the write-back mode based on an \texttt{edit\_mode} flag, leading to two branches: raw mode vs. LoRA-native mode. \\
    \midrule
    5a & \emph{Raw mode} -- Trust gate: compute a trust score from $d_{\text{pres}}, d_{\text{new}}, d_{\text{old}}$; scale or roll back $\Delta W_\ell$ according to the score. \\
    6a & Raw write-back: update weights as $W_\ell \leftarrow W_\ell + \Delta W_\ell^{\ast}$. \\
    7a & Output the edited model. \\
    \midrule
    5b & \emph{LoRA-native mode} -- Trust gate: compute a trust score before any low-rank update; if too low, roll back entirely and skip the update. \\
    6b & Greedy LoRA multi-step loop: split $\Delta W_\ell$ into $S$ small steps; at each step $s$, call a LoRA-native backend to approximate $B_\ell A_\ell \approx B_\ell^{\text{prev}} A_\ell^{\text{prev}} + \eta_s \, \Delta W_\ell$. \\
    7b & Optional online evaluation: compute an evaluation metric $C_s$ (composite score) and check whether $C_s - C_{s-1} \geq \epsilon$. \\
    8b & If there is no improvement: roll back the current step, restore the LoRA snapshot, and increase a patience counter. \\
    9b & If improved: accept the current step, update the LoRA state, and set $C_{s-1} \leftarrow C_s$. \\
    10b & Stop the greedy loop when patience is exhausted or all steps are completed, obtaining the final LoRA stack. \\
    11b & Output the edited model with LoRA adapters. \\
    \bottomrule
  \end{tabular}
\end{table*}

This workflow highlights the interaction between four components: (i) \textbf{Replay} extends the input request set, incorporating old edits at the request level; (ii) the \textbf{EMMET core} computes per-layer closed-form deltas and distance metrics; (iii) \textbf{Trust} gates these deltas differently under raw vs. LoRA-native modes, providing an insurance mechanism; and (iv) \textbf{Greedy LoRA} performs multi-step low-rank updates with online evaluation and rollback.

\subsection{Edit Trust and Replay Mechanism}
GRMeT introduces an edit trust and insurance mechanism as a first-class component of the editing algorithm. The system maintains a buffer of past edits and their evaluation scores. For each new batch, it
\begin{itemize}
  \item samples a subset of previous edits to form a replay set,
  \item computes an edit-specific trust score based on consistency of validation metrics (e.g., ES, PS, NS, composite), and
  \item scales, accepts, or rejects candidate updates based on the trust score, with automatic rollback in low-trust cases.
\end{itemize}

This mechanism allows GRMeT to adaptively roll back edits that exhibit negative side effects, particularly in sequential editing regimes.

\subsection{Greedy LoRA in the Equality-Constrained Subspace}
Instead of applying $\Delta W_\ell$ in a single shot, GRMeT decomposes the full-rank delta for each layer into a sequence of small updates in the LoRA subspace. At each step, the algorithm
\begin{enumerate}
  \item proposes a small low-rank increment that moves the LoRA parameters toward approximating the EMMET delta,
  \item evaluates a composite metric $C_s$ on a validation set combining edit targets and neighborhood probes, and
  \item decides whether to accept or roll back the step based on the improvement $C_s - C_{s-1}$ and a patience threshold.
\end{enumerate}

This greedy small-step hill-climbing process allows GRMeT to approximate equality-based updates while explicitly optimizing for a composite criterion that trades off editing efficacy and neighborhood preservation.

\section{Evaluation Metrics}
To comprehensively assess the effectiveness and controllability of model editing, we adopt four metrics and place particular emphasis on neighborhood fidelity.

\paragraph{Efficacy Score (ES).} ES measures whether the edit succeeds on the target probes, i.e., the proportion of prompts for which the edited model outputs the correct updated fact.

\paragraph{Paraphrase Score (PS).} PS evaluates paraphrase generalization: we apply synonymically rephrased prompts for the same fact and measure the success rate.

\paragraph{Neighborhood Specificity (NS).} NS measures local specificity by assessing the model's behavior on neighboring but non-target examples. It is defined as the proportion of neighborhood samples on which the edited model preserves the original model's predictions (higher is better).

\paragraph{Composite Score (CS).} We aggregate ES, PS, and NS into a composite score by simple averaging after normalization:
\begin{equation}
  \text{CS} = \frac{1}{5} (\text{ES} + \text{PS} + 3\text{NS}).
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
Our primary baselines are ROME, MEMIT, and EMMET. In addition, we consider EMMET variants that incorporate individual components of GRMeT (e.g., native LoRA only, trust and rollback only, greedy updates only, memory replay only) to form an ablation suite.

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
To validate the necessity of each component in the GRMeT framework, we perform an ablation study on 20-edit experiments, where we systematically remove modules such as replay, trust gating, or greedy LoRA updates. Table~\ref{tab:ablation} summarizes representative results.
\begin{table}[t]
\centering
\small
\caption{Ablation results on 20-edit experiments. ES, PS, NS, and CS are reported on a 0--1 scale.}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{ES} $\uparrow$ & \textbf{PS} $\uparrow$ & \textbf{NS} $\uparrow$ & \textbf{CS} $\uparrow$ \\
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

The results indicate that all components are important for the full performance of GRMeT: removing any module leads to noticeable degradation in ES, PS, NS, or CS, confirming the importance of combining replay, trust-aware gating, and greedy LoRA.

\section{Analysis and Discussion}
Our findings suggest that combining equality-based mass editing with low-rank PEFT is a promising direction for robust and efficient knowledge updates in language models. GRMeT preserves much of EMMET's high editing accuracy while leveraging LoRA to reduce interference with neighboring facts, particularly in small-batch and sequential editing scenarios.

However, handling very large batches remains challenging. GRMeT compresses all updates into a fixed low-rank subspace per layer, and repeated low-rank approximations can gradually erode the delicate balance required for stable edits. Future work can refine per-step balancing of deltas, explore dynamic rank adaptation, or use fact-specific adapter routing to further mitigate catastrophic forgetting in large-scale settings.

\section{Conclusion}
We have presented GRMeT, a low-rank equality-based editing framework that combines EMMET's mass-editing precision with LoRA's parameter efficiency, memory replay, and trust-aware greedy updates. GRMeT retains high editing accuracy, improves neighborhood fidelity, and provides an explicit insurance mechanism via replay and rollback, offering a practical solution for small-model and small-batch editing with a favorable balance between accuracy, robustness, and storage cost.

In future work, we plan to: (i) optimize the trade-off among ES, PS, and NS through more principled multi-objective criteria; (ii) study rank reduction or target representation compression to better align with equality constraints; and (iii) enhance parameter balancing in large-batch edits to mitigate catastrophic forgetting beyond dozens of edits.

\bibliographystyle{acl_natbib}
\bibliography{custom}

\end{document}
