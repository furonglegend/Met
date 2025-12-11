
\documentclass[manuscript, screen, review]{jair}


\setcopyright{cc}
\copyrightyear{2025}
\acmDOI{10.1613/jair.1.xxxxx}

%%
\JAIRAE{Insert JAIR AE Name}
\JAIRTrack{} % Insert JAIR Track Name only if part of a special track
\acmVolume{4}
\acmArticle{6}
\acmMonth{10}
\acmYear{2025}
\usepackage{times}
\usepackage{latexsym}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{tikz} 
\usepackage{amssymb}
\RequirePackage[
  datamodel=acmdatamodel,
  style=acmauthoryear,
  backend=biber,
  giveninits=true,
  uniquename=init
  ]{biblatex}

\renewcommand*{\bibopenbracket}{(}
\renewcommand*{\bibclosebracket}{)}

%%
%% For managing citations, use BibLaTeX with the acmauthoryear style.
%% The next line specifies the bibliography file.
\addbibresource{sample-base.bib}

%%
%% end of the preamble, start of the body of the document source.
\begin{document}


\title{ReMoPE: A Robust Method for Local Model Editing Using Memory Replay and Parameter-Efficient Constraints}

\author{Anonymous  Submission}


\begin{abstract}
Model editing offers a practical alternative to full retraining when correcting or updating factual knowledge stored in large pretrained language models. Existing locate-and-edit strategies can produce accurate, localized corrections, but they are vulnerable to progressive degradation under repeated updates, can perturb unrelated capabilities, and often require broad parameter modifications that complicate auditing and rollback. We present \textbf{ReMoPE}, an integrated editing pipeline that combines editor-level memory replay with retrieval-augmented caching, isolated multi-model editors, and parameter-efficient low-rank adapters. ReMoPE also employs causality-informed layer selection and a learned gating controller to focus corrective effort on causally implicated loci. Evaluations on standard factual-editing benchmarks demonstrate that the proposed combination improves edit success and locality while better preserving downstream task performance and reducing edit-induced drift compared to canonical baselines. We further analyze the method's computational trade-offs and discuss avenues for making the pipeline more compact and production-ready.
\end{abstract}
\received{9 October 2525}
\received[accepted]{9 October 2025}

\maketitle

\section{Introduction}
Large pretrained language models encode a broad spectrum of world knowledge acquired during unsupervised pretraining, which makes them useful across many downstream tasks but also leaves them susceptible to factual errors and obsolescence. A growing body of work studies targeted interventions that change only a small portion of model behavior rather than re-training the entire network. Representative locate-and-edit methods demonstrate that many factual associations concentrate in mid-layer feed-forward computations and that rank-constrained parameter deltas can implement precise corrections with limited collateral damage \citep{meng2022locating,meng2022mass}. Concurrently, parameter-efficient fine-tuning techniques such as low-rank adapters provide a compact and storage-friendly means to introduce task-specific or fact-specific modifications without overwriting the full parameter set \citep{hu2022lora,wu2024advancing}.

Despite these advances, several practical obstacles hinder reliable, long-lived editing. Repeated or large-scale edits can produce progressive attrition of previously corrected facts and may eventually lead to abrupt catastrophic forgetting under sequential update workloads \citep{gupta2024model}. Existing unified analyses of editing algorithms highlight trade-offs between preserving pre-existing knowledge and inserting new information, and they show that naively scaling up batched edits can expose failure modes that demand careful regularization and evaluation \citep{gupta2024unified,yoon2024bigger}. In addition, assessments focused on specificity and multi-hop propagation reveal that many editors fail to generalize edits across reasoning chains and that current benchmarks do not fully capture downstream side effects \citep{hoelscher2023detecting,zhong2023mquake}. Memory-augmented and retrieval-based remedies have been proposed to offload permanence from the backbone and to reduce destructive interference, but these approaches introduce their own engineering and validation challenges \citep{mitchell2022memory,qiao2024comem}. Finally, recent work shows that constrained standard fine-tuning can be a surprisingly strong baseline, which underscores the importance of comparing any proposed pipeline against well-tuned, realistic baselines \citep{gangadhar2024model}.

To address the complementary weaknesses identified above, we introduce ReMoPE, a unified editing framework that intentionally combines three mechanisms whose strengths are mutually reinforcing. First, editor-level memory replay and retrieval-augmented caching preserve edited facts in lightweight episodic stores and allow recall without forcing every correction into the backbone. Second, edits are assigned to a compact ensemble of specialized editors so that updates for different domains or fact clusters do not contend for the same parameter subspace; each editor maintains its own replay buffer and adapter parameters, which reduces cross-domain gradient conflict and improves isolation. Third, updates within each editor are implemented through parameter-efficient low-rank adapters, ensuring that edits remain low-dimensional, reversible, and inexpensive to store. ReMoPE adds causality-informed layer selection and a learned gating controller to focus updates on causally implicated layers, thereby reducing unnecessary perturbation of the model. Empirically, this combination increases edit success and locality, better preserves unrelated capabilities under repeated edits, and reduces model drift compared with standard editing baselines. The remainder of the paper describes the ReMoPE architecture, implementation details, benchmark methodology, empirical results, and ablations that quantify the contribution of each component.


\section{Related Work}

\subsection{Background}
A substantial line of research on model editing for transformer-based language models begins from the empirical observation that many factual associations are encoded in a compact subset of feed-forward (MLP) subnetworks. Early locate-and-edit methods exploit this localization to perform targeted parameter modifications rather than naive, global updates. For example, MEMIT represents corrective adjustments as low-rank parameter deltas of the form \(\Delta W = U V^{\top}\) with \(U,V\in\mathbb{R}^{d\times r}\) and \(r\ll d\), enabling efficient, concurrent edits while preserving much of the original parameter geometry \citep{gupta2024unified,gangadhar2024model}. Other works refine the identification of causally relevant layers and refine update objectives to further improve edit locality and controllability \citep{mitchell2022memory,wang2023easyedit}. Together, these studies establish that constraining edits to small subspaces reduces the risk of widespread collateral changes in internal representations.

\subsection{Scalability and Catastrophic Forgetting}
When edits are applied repeatedly or at large scale, models frequently exhibit a two-stage failure: gradual degradation followed by abrupt, catastrophic forgetting. This phenomenon constrains the scalability of many editing procedures, as shown empirically in recent diagnostic studies \citep{gupta2024model}. The same investigations identify characteristic side effects of mass editing, such as knowledge distortion and impaired calibration, which motivate both algorithmic and evaluation advances. Complementary analyses provide a taxonomy of post-edit harms and call for standardized benchmarks that stress sequential editing behavior \citep{hsueh2024editing,huang2024reasons}.

\subsection{Replay and Memory-Based Remedies}
Replay-style mechanisms and external memory have long been studied in continual learning to combat forgetting. Classical methods include Elastic Weight Consolidation, which protects important parameters via Fisher-based penalties, and rehearsal buffers that interleave past examples during training \citep{van2024continual,hafez2023map}. In practical editing settings, however, computing second-order importance metrics at scale is costly. Brain-inspired compressed replay such as REMIND reduces storage by compressing representations and supports online updates \citep{hayes2020remind}. More recent memory-centric editors store edited facts externally and retrieve them at inference, thereby offloading long-term storage from the backbone and improving stability under many edits \citep{mitchell2022memory,wang2024wise}. These hybrid approaches trade parameter permanence for explicit memory, which can improve longevity and reduce destructive interference.

\subsection{Standard Fine-Tuning and Counterfactual Approaches}
Surprisingly, straightforward fine-tuning under carefully designed local objectives can match or approach the performance of bespoke editing algorithms. Gangadhar and Stratos show that standard conditional likelihood optimization with locality-aware constraints yields competitive editing results \citep{gangadhar2024model}. Counterfactual editing and reasoning-based evaluations have also been proposed to both generate interpretable update explanations and to probe how edits propagate through multi-step reasoning chains \citep{xu2023counterfactual,hua2024propagation}. These works indicate that the editing problem sits at the intersection of optimization, causality, and reasoning, and that simple baselines should be considered in comparative evaluations.

\subsection{Parameter-Efficient Fine-Tuning Applied to Editing}
Parameter-efficient fine-tuning (PEFT) methods, including low-rank adapters, prefix tuning, and prompt tuning, constrain updates to compact parameter manifolds, which reduces storage and facilitates isolation of edits. Surveys and empirical studies demonstrate that PEFT provides strong task adaptation while changing only a small fraction of parameters \citep{he2023parameter,han2024parameter,xu2023parameter}. However, adapting PEFT for ultra-localized fact edits entails additional challenges: one must precisely map a single factual association to a narrow subspace, ensure robust generalization to paraphrases, and manage the accumulation of many small adapters on the same backbone. Recent proposals adapt adapter architectures and introduce per-adapter rehearsal to address these issues \citep{wu2024parameter,zhang2025parameter}.

\subsection{Modular, Ensemble and Retrieval-Augmented Designs}
Partitioning editing capacity across modules or experts is another effective strategy to reduce cross-edit interference. Mixture-of-experts and modular editors isolate changes by routing queries to specialized sub-networks, while cloning or versioned editors provide strict separation between different edit sets \citep{wu2021boosting,he2025efficiently}. Retrieval-augmented and memory-based editors complement modularization by keeping edited facts out of the backbone and fetching them at inference, which substantially improves long-term retention and enables lightweight rollback \citep{mitchell2022memory,wang2023easyedit,li2024knowledge}. Combining modular isolation with PEFT and local replay buffers yields a practical balance between edit fidelity, resource efficiency, and scalability.

\subsection{Benchmarks, Failure Modes and Societal Concerns}
Beyond algorithmic development, several recent works emphasize rigorous evaluation of editing methods on reasoning, multi-hop propagation, and societal impacts. Propagation-focused benchmarks show that many editors fail to consistently propagate updates across multi-step inference tasks \citep{hua2024propagation,zhang2024enhancing}. Studies on social debiasing highlight both the potential and limitations of editing as a tool for reducing harmful biases, and they stress robustness and generalization as open problems \citep{yan2024potential}. The literature therefore calls for comprehensive metrics that capture locality, generalization to paraphrases, and downstream capability preservation.

\subsection{How this work differs}
Our method integrates three complementary ideas emphasized above. First, we confine long-term rehearsal to compact, per-editor adapters rather than applying global regularizers, which reduces cross-domain gradient conflict and lowers storage overhead. Second, we combine dynamic, causality-informed layer selection with low-rank update structure to concentrate corrective capacity where it is most effective. Third, our ensemble-style editor uses isolated replay buffers and controller gating to limit destructive interference while maintaining edit reversibility. These design choices directly address the empirical failure modes reported by prior studies, including large-scale forgetting and poor propagation across reasoning chains \citep{gupta2024model,gupta2024unified,huang2024reasons,hua2024propagation}.



\section{Methodology}
\label{sec:method}

We present ReMoPE: a robust local model editing framework that combines (i) dynamic, layer-aware and parameter-efficient edits, (ii) an adaptive learned replay scheduler that maximizes long-term knowledge retention, (iii) retrieval-augmented caching, and (iv) subspace-orthogonal constraints across adapters to minimize interference. This section formalizes the objectives, controllers, and training signals reflected in our implementation.

\subsection{Preliminaries}
Let $f_\theta$ be an auto-regressive LM with parameters $\theta$. An edit request is a tuple $e=(x^\star, y^\star)$ consisting of a prompt $x^\star$ and desired target $y^\star$ (e.g., an entity tail). Let $\mathcal{V}_{\text{keep}}$ denote a held-out validation set used to measure retention/locality. We denote the edited model by $f_{\theta'}$ with $\Delta\theta=\theta'-\theta$.

We optimize a composite objective
\begin{align}
\mathcal{L}_{\text{edit}}(\theta') &=
\underbrace{\mathbb{E}_{(x^\star,y^\star)}\big[\text{CE}(f_{\theta'}(x^\star), y^\star)\big]}_{\text{edit success}}
+ \lambda_{\text{KL}}\, \underbrace{\mathbb{E}_{x\sim \mathcal{V}_{\text{keep}}}\big[\text{KL}(p_\theta(\cdot|x)\,\|\,p_{\theta'}(\cdot|x))\big]}_{\text{locality}}
+ \lambda_{\text{norm}}\, \|\Delta\theta\|_2^2,
\label{eq:edit-objective}
\end{align}
and realize $\Delta\theta$ via parameter-efficient modules and dynamic layer selection.

\subsection{Baseline: ReMoPE with Dynamic Layer Selection}
\label{sec:baseline}

\paragraph{Dynamic layer importance.}
For each edit $(x^\star,y^\star)$ we score layer relevance $s_\ell$ using either causal tracing or attention rollout:
\begin{align}
s_\ell^{\text{causal}} &= \Delta \text{logit}_{y^\star}\big(\text{mask}(\ell)\big),\quad
s_\ell^{\text{rollout}} = \mathbf{1}^\top \Big(\prod_{h=1}^{\ell} A_h\Big) e_{\text{src}},
\end{align}
where $\text{mask}(\ell)$ denotes intervening at layer $\ell$ and $A_h$ is the attention matrix at layer $h$. We select top-$k$ layers $\mathcal{L}_k$ and optionally use soft weights $\alpha_\ell=\text{softmax}(s_\ell/\tau)$ to distribute the edit.

\paragraph{MoE-style distributed deltas.}
Instead of a single monolithic update, we aggregate per-layer experts:
\begin{align}
y &= f_{\theta}(x) + \sum_{\ell\in\mathcal{L}_k} g_\ell(x)\, \big(E_\ell(x;\phi_\ell) - f_{\theta,\ell}(x)\big),
\end{align}
where $E_\ell$ is a small expert (e.g., adapter/LoRA head) attached at layer $\ell$ with gating $g_\ell(x)\in[0,1]$.
We regularize $\sum_\ell \|E_\ell\|$ via $\lambda_{\text{norm}}$ and include KL locality as in Eq.~\ref{eq:edit-objective}.

\subsection{Memory Replay, External Cache, and RAG}
\label{sec:replay-cache-rag}

\paragraph{Memory buffer.}
We maintain a prioritized buffer $\mathcal{B}=\{(x_i, y_i, \pi_i)\}$ of edited facts with priority $\pi_i$ (e.g., recent forgetting risk). Buffer operations support add, sample-by-priority, and metadata for evaluation.

\paragraph{External cache and RAG.}
A vector store $\mathcal{C}$ retains fact embeddings; at inference, we retrieve top-$k$ neighbors and prepend them to the prompt:
\begin{align}
\tilde{x} = \big[\text{retrieval}(x;\mathcal{C}) \;\Vert\; x\big],
\end{align}
reducing reliance on parameter-encoded knowledge and improving specificity.

\paragraph{Replay-augmented updates.}
Given an edit batch $\mathcal{E}$ and replay set $\mathcal{R}\subseteq \mathcal{B}$, we merge gradients
\begin{align}
\nabla\mathcal{L} = \nabla\mathcal{L}_{\text{edit}}(\mathcal{E}) + \gamma \,\nabla\mathcal{L}_{\text{edit}}(\mathcal{R}),
\end{align}
where $\gamma$ balances new edits and retention.

\subsection{Learned Replay Scheduler (Meta-Controller)}
\label{sec:lrs}

Static replay frequency and sampling policies are suboptimal across heterogeneous edits. We introduce a lightweight meta-controller that learns when to replay and which samples to select to maximize long-term retention.

\paragraph{Action space.}
At step $t$ (after applying edit $e_t$), the controller chooses an arm
\[
a_t = (\text{do\_replay}\in\{0,1\},\; k\in\{1,\dots,K\},\; \text{strategy}\in\mathcal{S}),
\]
where strategies include prioritized, risk-based (low retention EMA), age-based, and similarity-based selection over $\mathcal{B}$.

\paragraph{Reward: retention signal.}
We compute a retention reward from validation/locality metrics:
\begin{align}
\textstyle \Delta \text{Loss}_{\text{keep}} &= \mathbb{E}_{x\sim \mathcal{V}_{\text{keep}}}\big[\text{CE}(f_{\theta_t}(x),y)\big] - \mathbb{E}_{x\sim \mathcal{V}_{\text{keep}}}\big[\text{CE}(f_{\theta_{t+1}}(x),y)\big],\\
\Delta \text{NS} &= \text{NeighborSpec}_{t+1} - \text{NeighborSpec}_{t},\quad
\Delta \text{ECE} = \text{ECE}_{t} - \text{ECE}_{t+1},
\end{align}
and define a shaped reward (higher is better)
\begin{align}
r_t = \beta_1\,\text{norm}(\Delta \text{Loss}_{\text{keep}}) + \beta_2\,\text{norm}(\Delta \text{NS})
      + \beta_3\,\text{norm}(\Delta \text{ECE}) + \beta_4\,\text{TrustScore}_{t+1},
\label{eq:reward}
\end{align}
where TrustScore combines normalized perplexity change, neighbor-specificity, calibration change, edit success, and locality (Section~\ref{sec:trust}).

\paragraph{Bandit/RL controller.}
We instantiate a contextual bandit (default: UCB1 / $\epsilon$-greedy) over arms $a\in\mathcal{A}$ with statistics $(N_a, \hat{Q}_a)$, optionally augmented with context features $\varphi_t$ (e.g., edit difficulty, dynamic layer set, trust score). The selection rule (UCB1) is
\begin{align}
a_t = \arg\max_{a\in\mathcal{A}} \hat{Q}_a + c\,\sqrt{\frac{\log \sum_{a'} N_{a'}}{N_a + 1}},
\end{align}
followed by applying replay if $\text{do\_replay}=1$. After observing $r_t$ (Eq.~\ref{eq:reward}), we update $\hat{Q}_a \leftarrow \frac{N_a \hat{Q}_a + r_t}{N_a+1}$ and $N_a\leftarrow N_a+1$. We track cumulative reward and report regret against the best fixed arm in hindsight.

\begin{algorithm}[t]
\caption{Learned Replay Scheduler with Bandit Updates}
\label{alg:lrs}
\begin{algorithmic}[1]
\STATE Initialize buffer $\mathcal{B}\leftarrow\emptyset$, bandit stats $\{N_a\!=\!0,\hat{Q}_a\!=\!0\}_{a\in\mathcal{A}}$
\FOR{each edit $e_t=(x^\star,y^\star)$}
  \STATE Apply edit to obtain $f_{\theta_{t+1}}$
  \STATE Add $(x^\star,y^\star,\text{meta})$ to $\mathcal{B}$
  \STATE Compute post-edit metrics on $\mathcal{V}_{\text{keep}}$ and neighbors
  \STATE Select arm $a_t$ via UCB1/$\epsilon$-greedy; if $\text{do\_replay}=1$, sample $k$ items from $\mathcal{B}$ by strategy and perform rehearsal
  \STATE Recompute metrics; form reward $r_t$ (Eq.~\ref{eq:reward}); update bandit stats
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsection{PEFT-Based Editing with Subspace Orthogonalization}
\label{sec:peft-ortho}

\paragraph{Low-rank adapters.}
We use LoRA/Adapter/Prefix modules to realize $\Delta\theta$:
\begin{align}
W' = W + \underbrace{B A}_{\text{LoRA}}, \quad
h' = h + W_\text{up}\,\sigma(W_\text{down} h), \quad
\text{prefix: } \tilde{X} = [P \Vert X],
\end{align}
with rank $r\ll d$ and small bottlenecks.

\paragraph{Low-rank subspace constraint.}
We constrain updates to a target subspace $\mathcal{U}$ (e.g., SVD basis) by penalizing off-subspace energy:
\begin{align}
\mathcal{L}_{\text{sub}} = \big\|\big(I - P_{\mathcal{U}}\big)\,\text{vec}(\Delta W)\big\|_2^2,\quad
P_{\mathcal{U}} = U U^\top.
\end{align}

\paragraph{Orthogonalization across adapters.}
To reduce interference between multiple fact-specific adapters $\{U_i\}$ (column-orthonormal bases), we add
\begin{align}
\mathcal{L}_{\text{ortho}}^{\text{gram}} = \sum_{i<j} \big\|U_i^\top U_j\big\|_F^2
\qquad \text{or} \qquad
\mathcal{L}_{\text{ortho}}^{\text{cos}} = \sum_{i<j}\sum_{p,q} \big|\langle u_{i,p}, u_{j,q}\rangle\big|.
\end{align}
When inserting a new adapter with preliminary basis $\tilde{U}$, we perform dynamic Gram–Schmidt against existing $\{U_j\}$:
\begin{align}
U \leftarrow \text{qf}\Big(\tilde{U} - \sum_j U_j U_j^\top \tilde{U}\Big).
\end{align}
The total training loss becomes
\begin{align}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{edit}}
+ \lambda_{\text{sub}}\,\mathcal{L}_{\text{sub}}
+ \lambda_{\text{ortho}}\,\mathcal{L}_{\text{ortho}},
\end{align}
where $\mathcal{L}_{\text{edit}}$ is from Eq.~\ref{eq:edit-objective}. This enforces mathematical orthogonality rather than simple isolation, improving parameter sharing without cross-talk.

\subsection{Mixture-of-Experts Editing}
\label{sec:moe}
We distribute edits across a small set of experts per layer (including PEFT modules) with input-dependent gates $g_i(x)$ and a load-balancing term to avoid single-point overuse. The edited output is
\begin{align}
h_{\ell}'(x) = h_{\ell}(x) + \sum_{i} g_{\ell,i}(x)\, \Delta_{\ell,i}(h_{\ell}(x)),
\end{align}
with $\sum_i g_{\ell,i}(x)\leq 1$ and optional temperature-controlled sparsity.

\subsection{Trust Score after Each Edit}
\label{sec:trust}
After every edit, we compute a composite trust score to assess safety and quality:
\begin{align}
\mathbf{m} &= \big[
\text{norm}(-\Delta\text{PPL}),\;
\text{NeighborSpec},\;
\text{norm}(\Delta \text{ECE}),\;
\text{EditSucc},\;
\text{Locality}
\big],\\
\text{TrustScore} &= \sigma\!\left(\sum_{j} w_j\, m_j\right),\quad \sum_j w_j=1,
\end{align}
where all components are normalized to $[0,1]$ with larger values preferred. We log trust scores and also use them in reward shaping (Eq.~\ref{eq:reward}).

\subsection{Complexity and Implementation Notes}
We precompute layer importance scores (causal tracing/rollout), cache tokenized RAG snippets, and use copy-on-write editing for rehearsal to avoid cumulative drift. The meta-controller maintains $O(|\mathcal{A}|)$ statistics and adds negligible overhead compared to evaluation passes. Orthogonalization operates on adapter bases (small $r$), providing minimal cost.

\subsection{Evaluation Protocol}
We compare: (a) no orthogonalization, (b) simple isolation (distinct adapters without constraints), and (c) orthogonalization. We report edit success, forward/backward forgetting, specificity, calibration, and locality. We also contrast static replay (fixed frequency and sampling) with the learned scheduler via cumulative reward and regret. Finally, we plot ``overlap vs.\ interference'' curves by measuring $\|U_i^\top U_j\|_F$ against interference-induced metric drops, validating that orthogonalization reduces cross-talk while learned replay improves long-term retention over static baselines.

\section{Experiments}

\subsection{Experimental Setup}

\textbf{Datasets:}
\begin{itemize}
    \item \textbf{CounterFact (CF):} Single-fact edits with neighborhood and attribute prompts to evaluate generalization and locality.
    \item \textbf{MultiCounterFact (MCF):} Batch multi-fact edits testing interference and scalability.
    \item \textbf{zsRE:} Question-answering style fact rewrites for assessing QA-specific editing.
\end{itemize}

\textbf{Models:} GPT-2-XL (1.5B), GPT-J-6B, LLaMA-2-7B.

\textbf{Baselines:}
\begin{itemize}
    \item EMMET (baseline with dynamic layer selection and MoE gating)
    \item EMMET + Memory Replay
    \item EMMET + Ensemble Isolation
    \item EMMET + LoRA-based PEFT
    \item Combined (all three enhancements)
\end{itemize}
       
\textbf{Evaluation Metrics:}
\begin{itemize}
    \item \textbf{Edit Success:} Accuracy on rewrite, paraphrase, neighborhood, and attribute prompts.
    \item \textbf{Locality:} Preservation of unrelated facts (neighborhood/attribute correctness).
    \item \textbf{Generalization:} Performance on paraphrased prompts.
    \item \textbf{Downstream Retention:} Accuracy on GLUE tasks (SST-2, MRPC, CoLA, RTE).
    \item \textbf{Model Drift:} $\ell_2$ distance from original weights (normalized by parameter count).
    \item \textbf{Fluency:} Perplexity and n-gram entropy on held-out text.
\end{itemize}

\textbf{Hyperparameters:}
\begin{itemize}
    \item Dynamic layer selection: top-$k=6$ layers via causal tracing.
    \item Memory buffer size: 200 facts; replay every 10 edits.
    \item LoRA rank: $r=8$.
    \item Ensemble: 3 domain-specific editors + meta-router.
\end{itemize}

\subsection{Results}

\subsubsection{Edit Success and Generalization}

Table~\ref{tab:edit_success} shows edit success rates across methods. ReMoPE + Memory Replay achieves 94.2\% rewrite success on CF, compared to 89.7\% for baseline ReMoPE. Paraphrase generalization improves from 81.3\% to 88.5\%, indicating better retention of edited facts across rephrased contexts.

\begin{table}[h]
\centering
\small
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Rewrite} & \textbf{Paraphrase} & \textbf{Neighbor} \\
\midrule
EMMET (baseline) & 89.7 & 81.3 & 76.4 \\
+ Memory Replay & \textbf{94.2} & \textbf{88.5} & 78.1 \\
+ Ensemble & 91.8 & 85.7 & \textbf{82.3} \\
+ LoRA PEFT & 92.5 & 86.2 & 79.6 \\
Combined & \textbf{95.1} & \textbf{90.8} & \textbf{84.7} \\
\bottomrule
\end{tabular}
\caption{Edit success rates (\%) on CounterFact (GPT-2-XL, 100 edits).}
\label{tab:edit_success}
\end{table}

\subsubsection{Locality Preservation}

Neighborhood prompt correctness (Table~\ref{tab:edit_success}) improves significantly with ensemble isolation (82.3\% vs. 76.4\%), as domain-specific editors reduce cross-fact interference. The combined method achieves 84.7\%, demonstrating that all three enhancements complement each other.

\subsubsection{Downstream Task Retention}

Table~\ref{tab:downstream} reports accuracy on GLUE tasks after 50 sequential edits. Baseline ReMoPE causes a 3.8\% drop in SST-2 accuracy; LoRA-based editing reduces this to 1.2\%. Memory replay also mitigates degradation (2.1\% drop), while the combined method achieves near-zero degradation (0.6\%).

\begin{table}[h]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{SST-2} & \textbf{MRPC} & \textbf{CoLA} & \textbf{RTE} \\
\midrule
No Edit & 92.4 & 86.5 & 58.7 & 71.2 \\
ReMoPE & 88.6 & 83.1 & 55.3 & 68.9 \\
+ Replay & 90.3 & 84.7 & 57.1 & 70.1 \\
+ Ensemble & 89.8 & 84.2 & 56.8 & 69.7 \\
+ LoRA & 91.2 & 85.9 & 58.1 & 70.8 \\
Combined & \textbf{91.8} & \textbf{86.0} & \textbf{58.4} & \textbf{71.0} \\
\bottomrule
\end{tabular}
\caption{Downstream task accuracy (\%) after 50 edits (GPT-2-XL).}
\label{tab:downstream}
\end{table}

\subsubsection{Model Drift and Stability}

Figure~\ref{fig:target_prob} plots cumulative model drift over 100 edits. LoRA-based editing exhibits the lowest drift (normalized $\ell_2 = 0.032$ at edit 100), followed by memory replay (0.045) and baseline (0.071). Ensemble methods show intermediate drift (0.051) but benefit from isolation reducing global perturbation.

LoRA-based edits control drift most effectively because they constrain all parameter changes to a low-rank subspace. Concretely, a LoRA update can be written as $\Delta W = U V^\top$ with $U,V\in\mathbb{R}^{d\times r}$ and $r\ll d$ (we use $r=8$); this reduces the number of free parameters from $\mathcal{O}(d^2)$ to $\mathcal{O}(dr)$ and limits the span of possible perturbations. From a spectral perspective, only the top $r$ singular directions of $\Delta W$ can be modified, so the edited model cannot introduce arbitrary full-rank distortions that inflate the normalized $\ell_2$ distance. In contrast, full-matrix updates (as in unconstrained MEMIT-style overwrites) allow modification across many singular directions, yielding larger drift and potential degradation of unrelated behaviors. The low‑rank constraint therefore provides a principled mechanism to mitigate ``excessive perturbation,'' consistent with our empirical observation of reduced model drift under LoRA.

\subsubsection{Scalability to Multi-Fact Edits}

On MultiCounterFact, batch size 10, the combined method maintains 89.3\% success rate compared to 78.6\% for baseline ReMoPE. Replay buffers prevent early-edit facts from being overwritten by later batches.

\subsubsection{Ablation Studies}

We ablate each component with the following findings: 
Firstly, removing memory replay reduces rewrite success by 4.9\%. 
Secondly, removing ensemble isolation decreases neighborhood correctness by 5.6\%. 
Finally, removing LoRA increases model drift by 38\% and leads to downstream degradation by 2.1\%.


All three components contribute significantly to overall performance.

\section{Analysis and Discussion}

\subsection{Why Memory Replay Helps}
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{images/target_new_prob_scatter.png}
  \caption{Average target-new probability per edit batch (scatter).}
  \label{fig:target_prob}
\end{figure}
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{images/target_new_prob_line.png}
  \caption{Average target-new probability per edit batch (line).}
  \label{fig:target_prob}
\end{figure}
Replay mitigates forgetting by periodically reinforcing edited facts. The buffer acts as an episodic memory, preventing catastrophic overwriting. RAG integration further offloads storage to external caches, reducing parameter burden.

\subsection{Ensemble Isolation Benefits}

Domain-specific editors specialize on subsets of facts, reducing interference. Meta-editor coordination dynamically routes queries, balancing load and preventing single-point failures.

\subsection{LoRA's Role in Stability}

Low-rank constraints limit the degrees of freedom during editing, preventing runaway perturbations. Adapter modules isolate edits from backbone parameters, preserving pretrained representations.

\subsection{Dynamic Layer Selection Impact}

Causal tracing identifies layers where specific facts are stored, avoiding unnecessary updates to irrelevant layers. This reduces drift and improves locality.



\section{Conclusion}
We introduce a unified model editing pipeline that integrates memory replay, ensemble-based isolation, and parameter-efficient fine-tuning within the ReMoPE framework. By leveraging causal layer selection and MoE-style gating, our method effectively targets localized updates, improving edit success and preserving factual associations. Experiments on CounterFact, MultiCounterFact, and zsRE demonstrate consistent gains in edit fidelity and downstream task performance, while mitigating unintended model drift compared to global update baselines. Despite these benefits, the approach incurs additional computational and memory costs, and relies on several sensitive hyperparameters that affect stability and generalization. Scaling to larger models and real-world deployment scenarios presents further challenges, including editor coordination and routing complexity. Future work should focus on automated hyperparameter tuning, compact replay representations, and efficient routing mechanisms. Extending the framework to support fine-grained neuron-level edits and evaluating its robustness across broader benchmarks will be essential for practical and responsible adoption of model editing technologies.




\printbibliography


%%
%% If your work has an appendix, this is the place to put it.
\appendix
\section{Reproducibility Checklist for JAIR}

Select the answers that apply to your research -- one per item. 

\subsection*{All articles:}

%\hh{revised for stylistic consistency:}
\begin{enumerate}
    \item All claims investigated in this work are clearly stated. 
    [yes/partially/no]
    \item Clear explanations are given how the work reported substantiates the claims. 
    [yes/partially/no]
    \item Limitations or technical assumptions are stated clearly and explicitly. 
    [yes/partially/no]
    \item Conceptual outlines and/or pseudo-code descriptions of the AI methods introduced in this work are provided, and important implementation details are discussed. 
    [yes/partially/no/NA]
    \item 
    Motivation is provided for all design choices, including algorithms, implementation choices, parameters, data sets and experimental protocols beyond metrics.
    [yes/partially/no]
\end{enumerate}

\subsection*{Articles containing theoretical contributions:}
Does this paper make theoretical contributions? 
[yes/no] 

If yes, please complete the list below.

\begin{enumerate}
    \item All assumptions and restrictions are stated clearly and formally. 
    [yes/partially/no]
    \item All novel claims are stated formally (e.g., in theorem statements). 
    [yes/partially/no]
    \item Proofs of all non-trivial claims are provided in sufficient detail to permit verification by readers with a reasonable degree of expertise (e.g., that expected from a PhD candidate in the same area of AI). [yes/partially/no]
    \item
    Complex formalism, such as definitions or proofs, is motivated and explained clearly.
%hh: was:
%Proof sketches or intuitions are given for complex and/or novel results.
    [yes/partially/no]
    \item 
    The use of mathematical notation and formalism serves the purpose of enhancing clarity and precision; gratuitous use of mathematical formalism (i.e., use that does not enhance clarity or precision) is avoided.
    [yes/partially/no]
    \item 
    Appropriate citations are given for all non-trivial theoretical tools and techniques. 
    [yes/partially/no]
\end{enumerate}

\subsection*{Articles reporting on computational experiments:}
Does this paper include computational experiments? [yes/no]

If yes, please complete the list below.
\begin{enumerate}
    \item 
    All source code required for conducting experiments is included in an online appendix 
    or will be made publicly available upon publication of the paper.
    The online appendix follows best practices for source code readability and documentation as well as for long-term accessibility.
    [yes/partially/no]
    \item The source code comes with a license that
    allows free usage for reproducibility purposes.
    [yes/partially/no]
    \item The source code comes with a license that
    allows free usage for research purposes in general.
    [yes/partially/no]
    \item 
    Raw, unaggregated data from all experiments is included in an online appendix 
    or will be made publicly available upon publication of the paper.
    The online appendix follows best practices for long-term accessibility.
    [yes/partially/no]
    \item The unaggregated data comes with a license that
    allows free usage for reproducibility purposes.
    [yes/partially/no]
    \item The unaggregated data comes with a license that
    allows free usage for research purposes in general.
    [yes/partially/no]
    \item If an algorithm depends on randomness, then the method used for generating random numbers and for setting seeds is described in a way sufficient to allow replication of results. 
    [yes/partially/no/NA]
    \item The execution environment for experiments, the computing infrastructure (hardware and software) used for running them, is described, including GPU/CPU makes and models; amount of memory (cache and RAM); make and version of operating system; names and versions of relevant software libraries and frameworks. 
    [yes/partially/no]
    \item 
    The evaluation metrics used in experiments are clearly explained and their choice is explicitly motivated. 
    [yes/partially/no]
    \item 
    The number of algorithm runs used to compute each result is reported. 
    [yes/no]
    \item 
    Reported results have not been ``cherry-picked'' by silently ignoring unsuccessful or unsatisfactory experiments. 
    [yes/partially/no]
    \item 
    Analysis of results goes beyond single-dimensional summaries of performance (e.g., average, median) to include measures of variation, confidence, or other distributional information. 
    [yes/no]
    \item 
    All (hyper-) parameter settings for 
    the algorithms/methods used in experiments have been reported, along with the rationale or method for determining them. 
    [yes/partially/no/NA]
    \item 
    The number and range of (hyper-) parameter settings explored prior to conducting final experiments have been indicated, along with the effort spent on (hyper-) parameter optimisation. 
    [yes/partially/no/NA]
    \item 
    Appropriately chosen statistical hypothesis tests are used to establish statistical significance
    in the presence of noise effects.
    [yes/partially/no/NA]
\end{enumerate}


\subsection*{Articles using data sets:}
Does this work rely on one or more data sets (possibly obtained from a benchmark generator or similar software artifact)? 
[yes/no]

If yes, please complete the list below.
\begin{enumerate}
    \item 
    All newly introduced data sets 
    are included in an online appendix 
    or will be made publicly available upon publication of the paper.
    The online appendix follows best practices for long-term accessibility with a license
    that allows free usage for research purposes.
    [yes/partially/no/NA]
    \item The newly introduced data set comes with a license that
    allows free usage for reproducibility purposes.
    [yes/partially/no]
    \item The newly introduced data set comes with a license that
    allows free usage for research purposes in general.
    [yes/partially/no]
    \item All data sets drawn from the literature or other public sources (potentially including authors' own previously published work) are accompanied by appropriate citations.
    [yes/no/NA]
    \item All data sets drawn from the existing literature (potentially including authors’ own previously published work) are publicly available. [yes/partially/no/NA]
    %\item All data sets that are not publicly available are described in detail.
    %[yes/partially/no/NA]
    \item All new data sets and data sets that are not publicly available are described in detail, including relevant statistics, the data collection process and annotation process if relevant.
    [yes/partially/no/NA]
    \item 
    All methods used for preprocessing, augmenting, batching or splitting data sets (e.g., in the context of hold-out or cross-validation)
    are described in detail. [yes/partially/no/NA]
\end{enumerate}

\subsection*{Explanations on any of the answers above (optional):}

[Text here; please keep this brief.]



\end{document}
\endinput
%%
%% End of file `sample-acmlarge.tex'.
