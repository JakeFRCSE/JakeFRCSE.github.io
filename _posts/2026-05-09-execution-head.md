---
layout: post
title: Tracing Query-Conditioned Attention Heads in In-Context Learning
date: 2026-05-09
description: Research notes on identifying and interpreting query-conditioned attention heads in transformer language models.
tags: mechanistic-interpretability attention-heads circuit-tracing
categories: exploratory
thumbnail: /assets/img/execution-head/thumnail.png
---

Note: This post is temporarily uploaded for code rendering tests.

## TL;DR

If there are attention heads that execute a mapping by combining a task vector and a query, their behavior should not be fixed to a single answer. When the task context is fixed but the query changes, such heads should shape the model’s prediction based on the query. Conversely, when the query is fixed, changing the strength of the task signal, such as by varying the number of examples, should affect how strongly the task mapping is applied. To identify these execution-head candidates, I used logit attribution and activation patching to trace which heads influence the final prediction. The results reveal a set of heads that respond sensitively to query changes, making them promising candidates for further analysis of task execution in in-context learning.


## Motivation

Task Vectors and Function Vectors suggest that a task, or function, can be represented by a compact vector inside a language model. However, these vectors often fail to generalize across prompt formats. This raises several questions: what information do these vectors contain, how are they constructed, and under what conditions do they successfully induce a specific function? 

On the other hand, in the IOI paper, the authors identify a circuit by investigating attention heads' inputs and outputs and explaining how information is read, combined, and routed to the final output from a head-level perspective. This attention-head-based view is helpful for identifying the mechanisms of attention heads and allows us to gain a more fine-grained understanding of their roles.

This view provides a starting point for investigating which heads are responsible for executing a mapping from the query and task context to the correct response, and how they do so. In this post, I investigate the existence of such execution-head candidates in in-context learning.

## Core Idea

Before searching for important heads, I first narrow down the relevant source of the model’s ICL behavior. I start by verifying that the model performs the task reliably, so that there is a meaningful prediction signal to analyze. I then localize this signal across Transformer Blocks and examine whether it is primarily written by attention layers rather than MLP layers. Once attention layers are shown to be important, I move to a head-level analysis to identify the specific heads whose contributions depend on the query and shift toward the appropriate target.

## Experimental Setup

### 1. Model and prompts

- Model: `meta-llama/Llama-3.2-3B`, loaded in bfloat16 dtype to prevent OOM.
- Prompt format: following the Function Vectors paper, I use the `Q:\nA:\n\n` format.
- Number of examples: `0`, `1`, `2`, `3`, `4`, and `5`.

To make the n-shot settings comparable, I use the same query examples across different numbers of shots. The in-context examples are also constructed from the same ordered pool, so that the 0-shot through 5-shot prompts can be compared as progressively longer prefixes of the same example sequence. Counterfactual prompts are constructed by pairing prompts that share the same in-context examples but use different queries.

Here is an example comparing a 1-shot prompt and a 2-shot prompt.

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <img
    src="/assets/img/execution-head/counterfactual_query_1shot_2shots_combined.svg"
    alt="Comparison of 1-shot and 2-shot counterfactual query prompts"
    style="max-width: 100%; width: 900px; height: auto;"
  >
</div>

### 2. Dataset

Following the IOI circuit paper, I design the dataset so that token positions are easy to track during patching. The input-output pairs are extracted from the model vocabulary under constraints that make both the lowercase input and the capitalized output single-token words, so that the patching experiment can be conducted without having to worry about varying final-token positions.

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

### 3. Logit Attribution

Logit attribution is a convenient way to evaluate a component's output. It projects a component's output onto the difference between the unembedding vectors of the tokens of interest. By doing this, we can measure how much the output supports one answer token over the other. In this experiment, the attribution effect is measured by projecting the component's output onto the difference direction between the clean answer token and the counterfactual answer token. 

I run logit attribution over three activation types: accumulated_resid, decompose_resid, and stack_head_results. In all cases, attribution is measured at the final token by projecting the relevant residual-stream contribution onto the correct-minus-incorrect logit direction.

The attribution settings are:

- `accumulated_resid`: measures how the accumulated residual stream at the final token supports the clean answer.
- `decompose_resid`: measures the contribution of residual components, including embeddings, attention outputs, and MLP outputs.
- `stack_head_results`: measures the contribution of each attention head’s output at the final token.
- `paired_flip_logit_diff`: measures whether each head’s contribution flips with the counterfactual query and supports the currently correct answer.
- `paired_static_logit_diff`: measures whether each head has a fixed token-specific or side-specific preference that does not flip with the query.

### 4. Activation Patching

Activation patching is a way to evaluate whether a component is causally important for the model’s prediction. It replaces a component’s activation in the counterfactual run with the corresponding activation from the clean run, and measures how much this intervention shifts the model back toward the clean answer. In this experiment, the patching effect is measured by how much the patched run recovers the clean final logit difference from the counterfactual final logit difference.

The patching effect is computed as:

$$
\frac{\ell_i^{\mathrm{patched}}-\ell^{\mathrm{counterfactual}}}{\ell^{\mathrm{clean}}-\ell^{\mathrm{counterfactual}}}
$$

where $\ell^{\mathrm{clean}}$, $\ell^{\mathrm{counterfactual}}$, and $\ell_i^{\mathrm{patched}}$ denote the logit difference between the clean answer token and the counterfactual answer token in the clean, counterfactual, and patched runs, respectively. A score close to 1 indicates that patching the component largely recovers the clean behavior, while a score close to 0 indicates little causal effect.

I run activation patching over several activation types: `resid_pre`, `block_every`, `attn_head_out_all_pos`, `attn_head_out_last_pos`, and `attn_head_all_pos_every`.

The patching settings are:

- `resid_pre`: patch the residual stream before each transformer block.
- `block_every`: patch each major block component, including the residual stream, attention output, and MLP output.
- `attn_head_out_all_pos`: patch each attention head output across all token positions.
- `attn_head_out_last_pos`: patch each attention head output only at the final token position.
- `attn_head_all_pos_every`: patch each attention head’s output, query, key, value, and attention pattern across all token positions.

### 5. Validation

After identifying causally important heads, I validate whether the same heads remain important on held-out data. The held-out data are not used in the initial activation patching analysis. In particular, I compare positive-effect heads, negative-effect heads, and random heads to test whether the effects of the positive/negative heads are robust rather than due to chance.

## Results

### 1. Accuracy Across N-Shot Settings

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <img
    src="/assets/img/execution-head/average_logit_diff_by_n_shot.png"
    alt="Average logit difference and accuracy by number of ICL examples"
    style="max-width: 100%; width: 900px; height: auto;"
  >
</div>

While the accuracy saturates at the 2-shot setting, the average logit diff keeps increasing.

### 2. Logit Attribution

Here, I present the results of 5-shot setting because the logit diff is the highest and possibly shows the most robust figures.

- `accumulated_resid`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/accumulated_residual_logit_diff.html"
    title="Accumulated residual logit difference"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

Hmm... The logit diff keeps increasing, starting from the layer 13. But it's still hard to tell which layer is the most important for the logit diff. Let's see `decompose_resid`.

- `decompose_resid`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/decomposed_residual_logit_diff.html"
    title="Decomposed residual logit difference"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

Here, we can see that the 27_attn_out have the biggest logit diff! Also, overally, attn_outs are bigger than mlp_outs.

- `stack_head_results`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/per_head_logit_diff.html"
    title="Per-head logit difference"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

L27H02 looks the most important! Maybe this is the reason that the 27_attn_out showed the biggest logit diff. This time, let's find if the head is sensitive to the query change.

- `paired_flip_logit_diff` and `paired_static_logit_diff`

<div style="display: flex; gap: 1rem; align-items: stretch; flex-wrap: wrap; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/paired_flip_logit_diff.html"
    title="Paired flip logit difference"
    style="flex: 1 1 420px; min-width: 0; height: 520px; border: 0;"
    loading="lazy"
    scrolling="no"
  ></iframe>
  <iframe
    src="/assets/html/execution-head/paired_static_logit_diff.html"
    title="Paired static logit difference"
    style="flex: 1 1 420px; min-width: 0; height: 520px; border: 0;"
    loading="lazy"
    scrolling="no"
  ></iframe>
</div>


### 3. Activation Patching

### 4. Validation on Held-Out Data



## Interpretation



## Limitations
