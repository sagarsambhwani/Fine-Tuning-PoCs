# 🎯 Module 04: Preference Alignment & Reinforcement Learning (DPO, GRPO, RLHF)

Post-training alignment teaches an LLM **how to reason, follow nuanced human intent, maintain safety, and avoid sycophancy or reward hacking**. This module explores the theoretical mathematics, loss functions, algorithms, and practical implementations of modern alignment.

---

## 1. Why SFT Is Not Enough: The Alignment Objective

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE POST-TRAINING DILEMMA                                 │
│                                                                                        │
│  Supervised Fine-Tuning (SFT)        Preference Alignment (DPO / GRPO / RLHF)          │
│  • Maximizes likelihood of tokens    • Shifts probability mass from bad to good        │
│  • Suffers from Exposure Bias        • Evaluates complete generated sequences          │
│  • Treats subtle errors equally to   • Explicitly penalizes hallucinations, toxic      │
│    catastrophic failures               responses, and invalid logical steps            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Classic RLHF: The 3-Stage Pipeline (InstructGPT / PPO)

```
[ Stage 1: SFT ] ──► [ Stage 2: Reward Model (RM) ] ──► [ Stage 3: PPO Policy Optimization ]
 Train base model on   Train scalar scoring model        Optimize Policy π_θ against RM
 instruction pairs.     on (Prompt, Chosen, Rejected).    with KL penalty against π_ref.
```

### A. Bradley-Terry Preference Model
Given prompt $x$ and pair of outputs $(y_w \succ y_l)$ where $y_w$ is preferred over $y_l$:

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + e^{-(r(x, y_w) - r(x, y_l))}}$$

Reward model loss:
$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l)) \right]$$

### B. PPO Optimization Objective with KL Constraint
$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\psi(x, y) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)) \right]$$

* **The KL Penalty**: Prevents the policy $\pi_\theta$ from drifting too far from reference model $\pi_{\text{ref}}$, avoiding **reward hacking** (exploiting loopholes in the neural reward model).
* **The 4-Model Memory Burden**: PPO requires keeping **4 models simultaneously in GPU memory**:
  1. Policy Actor ($\pi_\theta$) — Active training
  2. Critic / Value Model ($V_\phi$) — Active training
  3. Reference Policy ($\pi_{\text{ref}}$) — Frozen
  4. Reward Model ($r_\psi$) — Frozen

---

## 3. Direct Preference Optimization (DPO)

*Reference: Rafailov et al., 2023 ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model")*

DPO bypasses training an explicit reward model and eliminates PPO's reinforcement learning loop by reparameterizing the Bradley-Terry preference model directly in terms of the language model policy.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               DPO VS PPO ARCHITECTURE                                  │
│                                                                                        │
│  PPO:  [Actor] ──► Generate (Rollout) ──► [Reward Model] ──► [Critic] ──► PPO Step    │
│        (Unstable, complex RL hyperparams, high VRAM)                                   │
│                                                                                        │
│  DPO:  [Dataset (x, y_w, y_l)] ──► [Forward Pass π_θ & π_ref] ──► Simple BCE Loss     │
│        (Stable, exact closed-form gradient, 2 models only)                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. Mathematical Derivation
The optimal policy under the KL-regularized reward maximization objective is:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)$$

Rearranging for the reward function $r(x, y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting $r(x, y)$ into the Bradley-Terry preference model ($Z(x)$ cancels out):

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### B. Understanding the $\beta$ Parameter
* $\beta \in [0.1, 0.5]$: Controls the strength of the KL penalty against $\pi_{\text{ref}}$.
* **Higher $\beta$**: Stricter adherence to $\pi_{\text{ref}}$ (more conservative, prevents degeneration).
* **Lower $\beta$**: Stronger preference optimization (can cause mode collapse or length exploitation if too low).

### C. Custom DPO Loss Implementation in Pure PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,    # log π_θ(y_w | x)
        policy_rejected_logps: torch.FloatTensor,  # log π_θ(y_l | x)
        reference_chosen_logps: torch.FloatTensor, # log π_ref(y_w | x)
        reference_rejected_logps: torch.FloatTensor,# log π_ref(y_l | x)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. Compute log ratios
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # 2. Compute scaled logits: β * (log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l)))
        logits = self.beta * (pi_logratios - ref_logratios)

        # 3. Compute DPO loss with optional label smoothing
        if self.label_smoothing > 0.0:
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # 4. Implicit rewards for tracking and evaluation
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
```

---

## 4. Modern Alignment Variants: KTO, ORPO & SimPO

| Algorithm | Key Innovation | Reference Model Required? | Data Requirement |
| :--- | :--- | :--- | :--- |
| **DPO** | Reparameterizes Bradley-Terry into policy loss | **Yes** ($\pi_{\text{ref}}$) | Pairwise $(x, y_w, y_l)$ |
| **KTO** (*Kahneman-Tversky Opt.*) | Maximizes utility under human prospect theory (loss aversion) | **Yes** ($\pi_{\text{ref}}$) | **Unpaired** binary $(x, y, \text{True/False})$ |
| **ORPO** (*Odds Ratio Preference Opt.*) | Combines SFT + Odds Ratio penalty in a single training step | **No** (Monolithic) | Pairwise $(x, y_w, y_l)$ |
| **SimPO** (*Simple Preference Opt.*) | Eliminates reference model and adds length-normalized margin $\gamma$ | **No** | Pairwise $(x, y_w, y_l)$ |

---

## 5. Group Relative Policy Optimization (GRPO) — The DeepSeek-R1 Paradigm

*Reference: DeepSeek-AI, 2025 ("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning")*

GRPO is the breakthrough reinforcement learning algorithm behind DeepSeek-R1 and modern reasoning models (o1-like reasoning).

```
                  GRPO TRAINING FLOW (DeepSeek-R1)
                  
                  Prompt: "Solve: 2x + 5 = 15"
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       Candidate o_1      Candidate o_2      Candidate o_3 (G samples)
       "x = 5"            "x = 10"           "x = 5"
            │                  │                  │
      [Rule Verifier]    [Rule Verifier]    [Rule Verifier]
         Reward: 1.0        Reward: 0.0        Reward: 1.0
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
        Normalize Advantage across Group G:
        A_i = (r_i - mean({r_1..r_G})) / std({r_1..r_G})
                               │
                               ▼
          Update Policy π_θ with Clipped Objective
```

### A. Why GRPO Replaces PPO
1. **Critic-Free Architecture**: PPO requires a Critic model (equal in size to the Actor) to estimate baseline values $V(s)$. GRPO eliminates the Critic completely, reducing training VRAM by $\approx 50\%$.
2. **Group Normalized Advantage**: For each query $q$, GRPO generates a group of $G$ outputs $\{o_1, o_2, \dots, o_G\}$. The baseline is the mean reward of the group:
   $$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$
3. **Verifiable Rule-Based Rewards**: Instead of an uncalibrated neural reward model (which suffers from reward hacking), GRPO uses deterministic verifiers:
   * **Math Verifier**: SymPy / exact string equivalence.
   * **Code Verifier**: Execution pass rate in Python sandbox.
   * **Format Verifier**: Enforcing `<think>...</think><answer>...</answer>` tags.
4. **Emergent Behaviors**: Under pure rule-based GRPO, models naturally develop **long chain-of-thought (CoT), self-reflection, backtracking ("Wait, let me rethink..."), and verification loops**.

---

## 6. Process Reward Models (PRMs) vs Outcome Reward Models (ORMs)

```
Outcome Reward Model (ORM):
[ Step 1: 2x = 10 ] ──► [ Step 2: x = 5 ] ──► [ Final Answer: 5 ] ──► Score: +1.0

Process Reward Model (PRM):
[ Step 1: 2x = 10 ] ──► Score: +1.0
[ Step 2: x = 5  ] ──► Score: +1.0  <-- Verifies EVERY intermediate step!
```

* **ORM Limitation**: An LLM can arrive at the correct answer through flawed logic (false positive). ORMs cannot penalize intermediate hallucinated steps.
* **PRM Advantage**: Provides step-level credit assignment, powering test-time search algorithms like **Monte Carlo Tree Search (MCTS)** and **Best-of-$N$ Beam Search**.

---

## 🎯 Top Interview Q&A on Alignment & RL

### Q1: What is the "Length Bias / Verbosity Exploitation" problem in DPO, and how is it solved?
**Answer**:
DPO relies on token log-probabilities summed over sequence length: $\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$. Because longer responses naturally accumulate more negative log-likelihood values, DPO can be tricked into favoring excessively wordy or verbose responses simply because length acts as a proxy for detail.
* **Solutions**:
  1. **SimPO / Length Normalization**: Divide sequence log-probability by length $|y|$: $\frac{1}{|y|} \log \pi(y \mid x)$.
  2. **Conservative DPO (cDPO)**: Assumes a label noise rate in preferences.
  3. **Length-Penalized Reward**: Explicitly subtract a length penalty during data generation.

### Q2: Why did DeepSeek-R1 choose GRPO over DPO or standard PPO?
**Answer**:
1. **DPO is Off-Policy & Static**: DPO optimizes over static offline datasets. It cannot explore new reasoning trajectories or discover self-correcting reasoning chains not present in the dataset.
2. **PPO is Memory-Prohibitive**: Scaling PPO across a 671B MoE architecture with a dedicated Critic model would require double the GPU infrastructure.
3. **GRPO Enables On-Policy Exploration with Verifiable Ground Truth**: GRPO samples dynamically on-policy, evaluates outputs with strict zero-noise verifiers (unit tests, math checks), and computes relative advantage without a Critic model.
