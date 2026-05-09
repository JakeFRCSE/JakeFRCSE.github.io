---
layout: post
title: Emergent Attention Pattern
date: 2026-05-09
description: Research notes on identifying and interpreting emergent attention patterns in transformer language models.
tags: mechanistic-interpretability attention-patterns
categories: exploratory
thumbnail: /assets/img/template_error.png
---

## TL;DR

Activation patching is conducted on Llama-3.2-3B increasing the number of examples from 0 to 1, 3, and 5. A set of heads with high causal logit effect on the output logit difference are identified, some of whom show emergent attention patterns that become distinct as the number of examples increases.

## Motivation

Task Vectors and Function Vectors showed that a task, or a function can be represented as a single vector. However, this representation is not generalizable to other formats of prompts. What information do the vectors contain? How are they constructed? What's going on inside the model when we inject the vectors? In this post, I investigate the attention heads that are responsible for the task execution by adapting the activation patching method to answer these questions.

## Core Idea

The main idea starts from the idea of the task vectors paper; according to the hypothesis class view, the model first maps the examples $\mathbf{S}$ into a parameter vector $\theta$ and then applies the rule of the vector to the query $x$. The paper investigated the parameter vectors in the activation space, whereas the function vectors paper constructed the parameter vectors from the output of a set of attention heads and suggested that this method is more effective. These approaches are related to the output of component, not to the construction mechanism.

If we understand the construction and triggering mechanisms of the function vectors, we would be able to build more refined and robust function vectors, enabling generalized application if the vectors.

To do this, the main goal of this project is to identify and explore the `execution head`, which formulate the function vectors. In terms of the computation model of attention head, we can expect the query and key to be the query $x$ and the parameter vector $\theta$.

## Experimental Setup

### 1. Model and prompts

- Model: `meta-llama/Llama-3.2-3B`
- Dataset is synthetized.
- Prompt format: Following the function vector paper, `Q:\nA:\n\n` format is used.
- Number of examples: `0 1 3 5`

### 2. Dataset

Following the IOI circuit paper, the prompt format used in this project is carefully designed. To be more specific, the data to insert into the format is extracted from the model's vocabulary following the conditions listed below, which allow us to facilitate patching experiments and tracking the token positions of our interest.

- Empty tokenizer pieces are filtered.
- Tokens with empty body after removing Ġ are filtered.
- The word body must start with a lowercase alphabetic letter.
- The capitalized token must also exist in the tokenizer vocabulary.
- Input must match ^[a-z]+$.
- Output must match ^[A-Z][a-z]+$.
- Input and output must each be at least 5 characters.
- Output must equal input.capitalize().
- Input must be in the English frequency word list.
- Input must tokenize to exactly one token without leading whitespace.
- Output must tokenize to exactly one token without leading whitespace.
- Input with leading whitespace must tokenize to exactly one token.
- Output with leading whitespace must tokenize to exactly one token.
- Duplicate (input, output) pairs are filtered.

### 3. Activation Patching

Following the ARENA tutorial 1.4, activation patching is conducted for `resid_pre`, `block_every`, `attn_head_out_all_pos`, `attn_head_out_last_pos`, and `attn_head_all_pos_every`. Activation patching is conducted by patching counterfacutal prompts and calculating the logit difference of the answer tokens of interest. For example, with the same exemplars in a prompt, there are two prompts of different queries. In this experiment, the goal is to see the logit difference between the original query and the counterfacutal query. I call this as clean and corrupt prompt following the convention, although the prompts are symmetric. After patching, outputs of the model, logit, will be projected into the difference vector, which is the difference direction between the corrupt token and the clean token. Projecting an activation into the difference vector is the same as projecting to each of the direction and take the differece.

Here are brief explanations of the experiments.

- resid_pre: patch the residual stream before each transformer block.
- block_every: patch each major block component: residual stream, attention output, and MLP output.
- attn_head_out_all_pos: patch each attention head output across all token positions.
- attn_head_out_last_pos: patch each attention head output only at the final token position.
- attn_head_all_pos_every: patch each attention head’s output, query, key, value, and attention pattern across all token positions.

### 4. Validation

After identifying causally important attention heads, the next step must be validating if the head are actually important in the held-out data. So the data that are not used in the activation patching are used for validation. Metrics for evaluating the heads are the following.

head effect metric: $$m_i = \frac{1}{N}\sum_{i=1}^{N}\frac{\ell_i^{\mathrm{patched}}-\ell^{\mathrm{corrupt}}}  {\ell^{\mathrm{clean}}-\ell^{\mathrm{corrupt}}}$$

head sign consistency metric: $$\frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[m_i > 0]$$

## Implementation

The implementation is available **[here](TODO_REPOSITORY_URL)**.

Summarize the main implementation choices. Mention libraries such as `TransformerLens`, `nnsight`, or custom hooks only if they are actually used. Include any compute limitations that affect the scope of the experiment.

## Results

### Attention visualization

Introduce the main attention heatmap or head-level visualization.

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <div style="max-width: 680px; width: 100%; display: flex; flex-direction: column;">
    <img
      src="/assets/img/template_error.png"
      alt="Attention pattern visualization"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: 0.5rem; font-size: 0.9em; color: var(--global-text-color-light);">TODO: Caption explaining the model, layer, head, token positions, and condition shown in the figure.</p>
  </div>
</div>

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
