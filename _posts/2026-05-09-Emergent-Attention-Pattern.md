---
layout: post
title: Emergent Attention Pattern
date: 2026-05-09
description: Research notes on identifying and interpreting emergent attention patterns in transformer language models.
tags: mechanistic-interpretability attention-patterns
categories: exploratory
thumbnail: /assets/img/template_error.png
---

Warning: This post is initial upload. Validation experiments are in progress.

## TL;DR

I study how task-execution behavior emerges in attention heads as the number of in-context examples increases. Using Llama-3.2-3B, I run activation patching on prompts with 0 to 5 examples and measure each component's causal effect on the output logit difference.

The main observation is that several heads with high causal effect also develop clearer attention patterns as more examples are added. This suggests that these heads may contribute not only to storing a task vector, but also to constructing or triggering it from the prompt structure.

## Motivation

Task Vectors and Function Vectors suggest that a task, or function, can be represented by a compact vector inside a language model. However, these vectors often fail to generalize across prompt formats. This raises several questions: what information do these vectors contain, how are they constructed, and what happens inside the model when such vectors are injected?

In this post, I investigate the attention heads involved in task execution. Instead of treating the function vector only as a final activation-space object, I use activation patching to identify the heads and internal components that causally affect the model's output on a simple in-context task.

## Core Idea

This project starts from the hypothesis-class view used in the Task Vectors paper. Under this view, the model maps the in-context examples $\mathbf{S}$ into a parameter vector $\theta$, then applies the rule represented by $\theta$ to the query $x$.

The Task Vectors paper studies this parameter vector in activation space. The Function Vectors paper instead constructs a task vector from the outputs of a small set of attention heads and shows that this construction can be more effective. Both approaches point to an important question: how is the vector constructed before it appears as an activation-space object?

If we understand the construction and triggering mechanisms of function vectors, we may be able to build more robust vectors that generalize better across prompt formats.

The goal of this project is therefore to identify and analyze possible `execution heads`, namely attention heads that help form or apply the function vector. From the perspective of an attention head, one natural hypothesis is that the query stream represents the test query $x$, while the key/value stream carries information about the inferred task parameter $\theta$.

## Experimental Setup

### 1. Model and prompts

- Model: `meta-llama/Llama-3.2-3B`, loaded in bfloat16 dtype to prevent OOM.
- Prompt format: following the Function Vectors paper, I use the `Q:\nA:\n\n` format.
- Number of examples: `0`, `1`, `2`, `3`, `4`, and `5`.

To make the n-shot settings comparable, I use the same query examples across different numbers of shots. The in-context examples are also constructed from the same ordered pool, so the 1-shot, 3-shot, and 5-shot prompts can be compared as progressively longer prefixes of the same example sequence.

### 2. Dataset

Following the IOI circuit paper, I design the dataset so that token positions are easy to track during patching. The input-output pairs are extracted from the model vocabulary under constraints that make both the lowercase input and capitalized output single-token words.

The filtering conditions are:

- Empty tokenizer pieces are removed.
- Tokens with empty bodies after removing `Ġ` are removed.
- The input word must start with a lowercase alphabetic character.
- The corresponding capitalized token must exist in the tokenizer vocabulary.
- The input must match `^[a-z]+$`.
- The output must match `^[A-Z][a-z]+$`.
- The input and output must each contain at least 5 characters.
- The output must equal `input.capitalize()`.
- The input must appear in an English frequency word list.
- The input must tokenize to exactly one token without leading whitespace.
- The output must tokenize to exactly one token without leading whitespace.
- The input with leading whitespace must tokenize to exactly one token.
- The output with leading whitespace must tokenize to exactly one token.
- Duplicate `(input, output)` pairs are removed.

### 3. Activation Patching

Following ARENA tutorial 1.4, I run activation patching over several activation types: `resid_pre`, `block_every`, `attn_head_out_all_pos`, `attn_head_out_last_pos`, and `attn_head_all_pos_every`.

For each patching experiment, I construct a clean prompt and a counterfactual prompt. The two prompts share the same in-context examples but use different queries. The goal is to measure how much patching a component from the clean run into the counterfactual run shifts the model toward the clean answer. Here, I used $|N|=200$ prompts.

The patching effect is measured with the logit difference between the clean answer token and the counterfactual answer token. Equivalently, the model output is projected onto the difference direction between these two answer tokens.

The patching settings are:

- resid_pre: patch the residual stream before each transformer block.
- block_every: patch each major block component, including the residual stream, attention output, and MLP output.
- attn_head_out_all_pos: patch each attention head output across all token positions.
- attn_head_out_last_pos: patch each attention head output only at the final token position.
- attn_head_all_pos_every: patch each attention head’s output, query, key, value, and attention pattern across all token positions.

### 4. Validation

After identifying causally important heads, I validate whether the same heads remain important on held-out data. The held-out data are not used during the initial activation patching analysis.

I use two metrics to evaluate each head.

Head effect metric:

$$
\frac{1}{N}\sum_{i=1}^{N}\frac{\ell_i^{\mathrm{patched}}-\ell^{\mathrm{corrupt}}}{\ell^{\mathrm{clean}}-\ell^{\mathrm{corrupt}}}
$$

Head sign consistency metric:

$$
\frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[m_i > 0],
\quad \text{where } m_i \text{ is the head effect metric for data point } i
$$

## Results

### Attention Patching

Here, the results of 3-shot patching are presented.

Describe the qualitative pattern shown in the figure. State whether it appears consistently across examples or only under a narrow condition.

### Quantitative summary

Use a table when the result is easier to compare across models, layers, heads, or prompt conditions.

<div class="table-responsive" style="margin: 1rem 0;">
<table class="table table-bordered table-sm" style="margin-bottom: 0;">
  <caption style="caption-side: top; text-align: left; padding-bottom: 0.5rem; font-size: 0.9em; color: var(--global-text-color-light);">TODO: Short caption describing the metric and conditions.</caption>
  <thead>
    <tr>
      <th>Model</th>
      <th>Layer</th>
      <th>Head</th>
      <th>Condition</th>
      <th>Metric</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>`TODO_MODEL`</td><td>TODO</td><td>TODO</td><td>TODO</td><td>TODO</td></tr>
  </tbody>
</table>
</div>

Summarize the key numerical result and distinguish observed trends from causal claims.

## Interpretation

Explain what the result suggests about the model's computation.

Be explicit about alternative explanations. For example, the pattern may reflect a prompt artifact, token-position effect, dataset bias, or downstream value-vector behavior rather than a standalone mechanism.

## Limitations

List the main reasons the result should be interpreted cautiously.

- Limited model coverage: TODO
- Limited dataset or prompt diversity: TODO
- Correlation versus causation: TODO
- Sensitivity to tokenization or prompt format: TODO

## Reflection

Conclude with what this experiment clarified and what should be tested next.

Possible next steps include layer-wise analysis, cross-model comparison, causal head patching, value-vector analysis, or testing whether the pattern generalizes to a different task family.
