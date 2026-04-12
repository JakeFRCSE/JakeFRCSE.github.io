---
layout: post
title: Zero-Shot Inference and Relation Vectors
date: 2026-04-12
description: Applying the Function Vectors methodology to zero-shot inference and investigating why the relation vector fails on incorrectly answered examples.
tags: mechanistic-interpretability function-vectors
categories: exploratory
thumbnail: /assets/img/zero-shot_inference/patching_head_viz.png
---

# TL;DR

This post applies the methodology of the [Function Vectors](https://arxiv.org/pdf/2310.15213) paper to zero-shot inference. Instead of in-context learning (ICL) prompts, a zero-shot prompt format is used to derive a **relation vector** via activation patching. While the relation vector successfully restores correct answers in the correctly-answered subset (over 80% restoration), it fails to produce a comparable effect on the incorrectly-answered subset (below 40%), suggesting that the failure mode in zero-shot inference involves more than just a weak relation signal.

# Motivation

The [Function Vectors](https://arxiv.org/pdf/2310.15213) paper investigated the function vectors of in-context learning (ICL). The paper showed that specific attention heads carry task-relevant information during ICL, and that a compact vector summarizing these heads' outputs can trigger task execution when injected into the residual stream. However, how LLMs solve problems in zero-shot settings, where no in-context examples are provided, is under-explored.

# Core Idea

This experiment follows the same methodology as the paper but uses a different prompt format. Instead of few-shot ICL prompts, a zero-shot prompt is used:

`Relation: {relation}\nInput: {input_val}\nOutput:`

The goal is to investigate whether a **relation vector** derived from zero-shot prompts enforces correct answers in zero-shot inference. Throughout this post, I use the term "relation vector" to distinguish this zero-shot analogue from the original function vector.

# Preliminary Research

As a preliminary experiment, I used the antonym dataset from the Function Vectors codebase to evaluate how well LLMs perform zero-shot inference. The results showed that most of the failed cases were instances of **input repeat**, where the model simply echoed the input word instead of producing its antonym. This suggests that the model's default fallback behavior is to repeat the input.

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <div style="max-width: 600px; width: 100%; display: flex; flex-direction: column;">
    <img
      src="/assets/img/zero-shot_inference/accuracy_plot.png"
      alt="Prediction accuracy by relation condition"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: 0.5rem; font-size: 0.9em; color: var(--global-text-color-light);">Prediction accuracy across three relation conditions (Antonym, None, Repeat). With the "antonym" relation, the model produces the correct output 53.5% of the time but still falls back to input repeat in 16.8% of cases. Under "none" and "repeat" relations, input repeat dominates at around 80%.</p>
  </div>
</div>

# Experiment Details

## 1. Dataset Split

Following the original paper, the dataset is split into two subsets: one consisting of examples the model answers correctly (used to derive the relation vector) and another consisting of incorrectly answered examples. Also following the paper, the number of prompts used for the clean run and the corrupted run are 100 and 25, respectively.

## 2. Patching

To derive the relation vector, I employed the activation patching method. The **clean run** uses the zero-shot prompt as specified above. The **corrupt run** replaces the relation word "antonym" with "none." In the preliminary experiment, the "none" relation triggered the fallback mode (input repeat), so replacing "antonym" with "none" effectively removes the relation signal while preserving the rest of the prompt structure.

After patching, the Causal Indirect Effect (CIE) is calculated for each attention head. This result is consistent with the original paper: attention heads with strong CIE appear in the middle layers of the model. The relation vector is then constructed by summing the mean output vectors of the top-10 heads (marked with red borders below), projected into the hidden state space through their corresponding attention head output matrices.

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <div style="max-width: 600px; width: 100%; display: flex; flex-direction: column;">
    <img
      src="/assets/img/zero-shot_inference/patching_head_viz.png"
      alt="CIE heatmap of attention heads"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: 0.5rem; font-size: 0.9em; color: var(--global-text-color-light);">CIE heatmap across all attention heads. The top-10 causally important heads are highlighted with red borders and are concentrated in the middle layers (roughly layers 12–17).</p>
  </div>
</div>

## 3. Intervention

Intervention experiments were conducted layer-wise in three settings:

1. **Normal intervention** (`intervention_fig`): For the correctly answered subset, the relation is corrupted (replaced with "none") and the relation vector is added. This is the natural validation setting — it tests whether the relation vector can restore correct behavior when the relation signal has been removed.

2. **Enforce intervention** (`intervention_enforce_fig`): For the incorrectly answered subset, the relation is kept intact and the relation vector is added. This setting tests whether the model's failure on these examples is due to a weak relation signal that can be strengthened.

3. **Restore intervention** (`intervention_restore_fig`): For the incorrectly answered subset, the relation is corrupted and the relation vector is added. This setting tests whether the failure can be corrected when the original relation signal is removed and replaced with the relation vector.

# Results

Across the three intervention configurations, the normal intervention showed high restoration of correct answers (over 80%). However, neither the restore setting nor the enforce setting reached above 40% correction on the incorrectly answered subset.

<div style="display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap; margin: 1.5rem 0;">
  <div style="flex: 1 1 100%; display: flex; flex-direction: column;">
    <strong>Normal intervention (correctly answered subset, relation corrupted)</strong>
    <img
      src="/assets/img/zero-shot_inference/intervention_fig.png"
      alt="Normal intervention result"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: 0.5rem; font-size: 0.9em; color: var(--global-text-color-light);">The relation vector restores correct output prediction up to ~80% around the middle layers, while input prediction drops correspondingly.</p>
  </div>
</div>

<div style="display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap; margin: 1.5rem 0;">
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>Enforce intervention (incorrectly answered subset, relation intact)</strong>
    <img
      src="/assets/img/zero-shot_inference/intervention_enforce_fig.png"
      alt="Enforce intervention result"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem; font-size: 0.9em; color: var(--global-text-color-light);">Adding the relation vector on top of the intact relation signal yields only ~35% peak correction. The effect is much weaker than the normal intervention.</p>
  </div>
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>Restore intervention (incorrectly answered subset, relation corrupted)</strong>
    <img
      src="/assets/img/zero-shot_inference/intervention_restore_fig.png"
      alt="Restore intervention result"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem; font-size: 0.9em; color: var(--global-text-color-light);">Replacing the corrupted relation signal with the relation vector reaches only ~25% peak correction, confirming that the failure mode is not simply a missing relation signal.</p>
  </div>
</div>

Consistent with the original paper, the intervention effect peaks around layer $$L/3$$ (approximately layer 9–10 for Llama-3.2-3B with $$L = 28$$), indicating that the relation signal is most effective when injected in the early-to-middle layers of the model.

# Implication

One notable observation is that the relation vector was not effective enough on the incorrectly answered subset to produce an impact comparable to what was observed in the correctly answered subset. Since the relation signal alone is not the key factor behind these failures, the underlying cause is likely a more complex problem. A natural next step can be to investigate the failure mode further, conjecturing that it may be related to the model's lack of "understanding" of the input word itself.

# Limitations

This experiment is limited by the following factors:

- Due to limited computational resources, all experiments are conducted on a single task (antonym) and a single model (Llama-3.2-3B). Extending the analysis to other relation types and larger models would strengthen the generality of these findings.
- Answer prediction is evaluated based on the first token of the answer word, not the entire word.

<div class="appendix-small" markdown="1">

# Appendix

This appendix provides the full mathematical notation for the methods used in this post. Appendix A defines the Causal Indirect Effect (CIE) used to identify causally important attention heads, and Appendix B defines the relation vector constructed from those heads.

## Appendix A. Causal Indirect Effect (CIE) — Full Notation and Explanation

### A.1. Model Setup

We consider an autoregressive transformer language model:

- Model: $f$
- Input prompt: $p$
- Vocabulary: $V$

The model outputs a next-token distribution:

$$
f(p) \in \Delta^{|V|}
$$

For a token $y \in V$:

$$
f(p)[y] \in [0,1]
$$

This is the probability that the model predicts token $y$ given prompt $p$.

---

### A.2. Transformer Internal Notation

The model has $L$ layers.

At layer $\ell$, the hidden state at the final token is:

$$
h_\ell \in \mathbb{R}^d
$$

Each layer is computed as:

$$
h_\ell = h_{\ell-1} + m_\ell + \sum_{j=1}^{J} a_{\ell,j}
$$

where:

- $m_\ell$: MLP output at layer $\ell$
- $a_{\ell,j}$: contribution of attention head $j$ in layer $\ell$
- $J$: number of attention heads per layer

Each attention head output depends on the input prompt:

$$
a_{\ell,j}(p)
$$

---

### A.3. Zero-Shot Inference Setup

We define a relation $r$ (e.g., "antonym") with dataset:

$$
P_r = \{p_i^r\}
$$

Each prompt is a zero-shot prompt of the form:

$$
p_i^r = [\text{Relation: } r, \text{ Input: } x_i, \text{ Output:}]
$$

where:

- $r$: relation type (e.g., "antonym")
- $x_i$: input word
- $y_i$: correct answer (not shown in the prompt)

We only keep prompts where the model correctly predicts $y_i$.

---

### A.4. Corrupted (Uninformative) Prompts

We define a corrupted version by replacing the relation with an uninformative one:

$$
\tilde{p}_i^r = [\text{Relation: none}, \text{ Input: } x_i, \text{ Output:}]
$$

This removes the relation signal while keeping the prompt structure intact, triggering the model's fallback behavior (input repeat).

---

### A.5. Mean Relation-Conditioned Activation

For each attention head $a_{\ell,j}$, compute:

$$
\bar{a}_{\ell,j}^r = \frac{1}{|P_r|} \sum_{p_i^r \in P_r} a_{\ell,j}(p_i^r)
$$

This is the average activation of head $(\ell,j)$ over clean prompts for relation $r$.

---

### A.6. Intervention (Activation Patching)

We run the model on corrupted prompt $\tilde{p}_i^r$, but replace one head's activation:

$$
a_{\ell,j} := \bar{a}_{\ell,j}^r
$$

This means that we inject the correct relation-specific signal into a corrupted prompt.

---

### A.7. Causal Indirect Effect (CIE)

The CIE is defined as:

$$
\mathrm{CIE}(a_{\ell,j} \mid \tilde{p}_i^r)
=
f(\tilde{p}_i^r \mid a_{\ell,j} := \bar{a}_{\ell,j}^r)[y_i]
-
f(\tilde{p}_i^r)[y_i]
$$

---

### A.8. Interpretation

This measures:

- the probability of the correct answer after intervention:

  $$
  f(\tilde{p}_i^r \mid a_{\ell,j} := \bar{a}_{\ell,j}^r)[y_i]
  $$

- the probability of the correct answer without intervention:
  $$
  f(\tilde{p}_i^r)[y_i]
  $$

So:

$$
\mathrm{CIE} = \text{(with correct signal)} - \text{(without signal)}
$$

It quantifies how much this attention head causally contributes to producing the correct answer.

---

### A.9. Intuition

- If CIE is large, this head carries task-relevant information.
- If CIE is near zero, this head does not contribute substantially to the task.

---

### A.10. Average Indirect Effect (AIE)

To aggregate over all corrupted prompts:

$$
\mathrm{AIE}(a_{\ell,j}) =
\frac{1}{|\tilde{P}_r|}
\sum_{\tilde{p}_i^r \in \tilde{P}_r}
\mathrm{CIE}(a_{\ell,j} \mid \tilde{p}_i^r)
$$

This identifies globally important heads for the relation.

---

## Appendix B. Relation Vector Formulation

After computing the average indirect effect (AIE) for every attention head, we select a set of the most causally important heads:

$$
A = \{\text{top attention heads ranked by AIE}\}
$$

Here, $A$ is the set of heads that appear to contribute most strongly to producing the correct answer for relation $r$.

---

### B.1. Relation-Conditioned Mean Activation of a Head

For each selected attention head $a_{\ell,j} \in A$, define its relation-conditioned mean activation for relation $r$ as:

$$
\bar{a}_{\ell,j}^r
=
\frac{1}{|P_r|}
\sum_{p_i^r \in P_r} a_{\ell,j}(p_i^r)
$$

This is the average output of head $(\ell,j)$ over clean prompts for relation $r$.

Interpretation:

- $a_{\ell,j}(p_i^r)$: output of head $(\ell,j)$ on one clean prompt
- $\bar{a}_{\ell,j}^r$: typical head output when the model is processing relation $r$

---

### B.2. Relation Vector Definition

The relation vector for relation $r$ is defined as the sum of these relation-conditioned mean head outputs over the selected causal heads:

$$
v_r
=
\sum_{a_{\ell,j} \in A} \bar{a}_{\ell,j}^r
$$

This follows the same construction as Equation (5) in the Function Vectors paper, adapted to the zero-shot setting.

---

### B.3. Interpretation of the Relation Vector

The intuition is:

- each important head in $A$ carries part of the relation-relevant signal
- summing their average outputs gives one compact vector
- this vector is meant to represent the relation in the residual stream

So $v_r$ is a compressed representation of the internal signal typically transported when the model processes relation $r$.

---

### B.4. Why This Sum Makes Sense

The formulation relies on the fact that attention head outputs are written in the residual-stream space.

The hidden state is defined as:

$$
h_\ell = h_{\ell-1} + m_\ell + \sum_{j=1}^{J} a_{\ell,j}
$$

Since each $a_{\ell,j} \in \mathbb{R}^d$ already lives in the same residual-stream space as $h_\ell$, their sum also lives in that same space.

Therefore:

$$
v_r \in \mathbb{R}^d
$$

and it can be directly added to hidden states.

---

### B.5. Relation Vector Intervention

Once $v_r$ is constructed, we test it by adding it to a hidden state at layer $\ell$:

$$
h_\ell' = h_\ell + v_r
$$

Equivalently, using the expanded residual-stream form:

$$
h_\ell'
=
h_{\ell-1} + m_\ell + \sum_{j=1}^{J} a_{\ell,j} + v_r
$$

---

### B.6. Model Output Under Relation Vector Intervention

If we intervene on prompt $p$ by adding $v_r$ at layer $\ell$, the resulting model behavior can be written as:

$$
f(p \mid h_\ell := h_\ell + v_r)
$$

This means:

- run the model on prompt $p$
- at layer $\ell$, replace the hidden state with $h_\ell + v_r$
- continue the forward pass and observe the new output distribution

---

### B.7. Practical Meaning

The full pipeline is:

1. Collect clean zero-shot prompts for relation $r$.
2. Compute mean activation $\bar{a}_{\ell,j}^r$ for each head.
3. Compute AIE for all heads using corrupted prompts.
4. Select top causal heads $A$.
5. Sum their relation-conditioned mean outputs to form:
   $$
   v_r = \sum_{a_{\ell,j}\in A} \bar{a}_{\ell,j}^r
   $$
6. Add $v_r$ to a hidden state during inference.
7. Measure whether the model now produces the correct answer for relation $r$.

---

### B.8. Compact Summary

The relation vector is not defined as an arbitrary learned vector.

It is constructed from actual model activations:

$$
v_r
=
\sum_{a_{\ell,j}\in A}
\left(
\frac{1}{|P_r|}
\sum_{p_i^r\in P_r} a_{\ell,j}(p_i^r)
\right)
$$

So it is:

- relation-specific
- activation-derived
- causally motivated, because the selected heads are chosen by AIE/CIE

</div>
