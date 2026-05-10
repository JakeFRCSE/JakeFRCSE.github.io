---
layout: post
title: Tracing Query-Conditioned Attention Heads in In-Context Learning
date: 2026-05-09
description: Research notes on identifying and interpreting query-conditioned attention heads in transformer language models.
tags: mechanistic-interpretability attention-heads circuit-tracing
categories: exploratory
thumbnail: /assets/img/execution-head/thumbnail.png
---

## TL;DR

If there are attention heads that execute a mapping by combining a task vector and a query, their behavior should not be fixed to a single answer. When the task context is fixed but the query changes, such heads should shape the model’s prediction based on the query. Conversely, when the query is fixed, changing the strength of the task signal, such as by varying the number of examples, should affect how strongly the task mapping is applied. To identify these execution-head candidates, I used logit attribution and activation patching to trace which heads influence the final prediction. The results reveal a set of heads that respond sensitively to query changes, making them promising candidates for further analysis of task execution in in-context learning.

## Motivation

[Task Vectors](https://arxiv.org/pdf/2310.15916) and [Function Vectors](https://arxiv.org/pdf/2310.15213) suggest that a task, or function, can be represented by a compact vector inside a language model. However, these vectors often fail to generalize across prompt formats. This raises several questions: what information do these vectors contain, how are they constructed, and under what conditions do they successfully induce a specific function?

On the other hand, in the [IOI paper](https://arxiv.org/pdf/2211.00593), the authors identify a circuit by investigating attention heads' inputs and outputs and explaining how information is read, combined, and routed to the final output from a head-level perspective. This attention-head-based view is helpful for identifying the mechanisms of attention heads and allows us to gain a more fine-grained understanding of their roles.

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <img
    src="/assets/img/execution-head/thumbnail.png"
    alt="Conceptual diagram of execution heads using task context and query inputs"
    style="max-width: 100%; width: 900px; height: auto;"
  >
</div>

In (a), the same task representation $\theta(S)$ and query $x_1$ are combined by the execution heads to produce $y_1$. In (b), the task representation is unchanged, but the query changes to $x_2$, so the output should change to $y_2$. In (c), replacing the query information tests whether the heads are truly query-sensitive rather than only carrying a fixed answer preference.

This view provides a starting point for investigating which heads are responsible for executing a mapping from the query and task context to the correct response, and how they do so. In this post, I investigate the existence of such execution-head candidates in in-context learning.

## Core Idea

Before searching for important heads, I first narrow down the relevant source of the model’s ICL behavior. I start by verifying that the model performs the task reliably, so that there is a meaningful prediction signal to analyze. I then localize this signal across Transformer Blocks and examine whether it is primarily written by attention layers rather than MLP layers. Once attention layers are shown to be important, I move to a head-level analysis to identify the specific heads whose contributions depend on the query and shift toward the appropriate target.

## Experimental Setup

### 1. Model and prompts

- Model: `meta-llama/Llama-3.2-3B`, loaded in bfloat16 dtype to prevent OOM.
- Prompt format: following the [Function Vectors paper](https://arxiv.org/pdf/2310.15213), I use the `Q:\nA:\n\n` format.
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

Following the [IOI circuit paper](https://arxiv.org/pdf/2211.00593), I design the dataset so that token positions are easy to track during patching. The input-output pairs are extracted from the model vocabulary under constraints that make both the lowercase input and the capitalized output single-token words, so that the patching experiment can be conducted without having to worry about varying final-token positions.

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

I run activation patching over several activation types: `resid_pre` and `attn_head_out_last_pos`.

The patching settings are:

- `resid_pre`: patch the residual stream before each transformer block.
- `attn_head_out_last_pos`: patch each attention head output only at the final token position.

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

The accuracy already saturates at the 2-shot setting, but the average logit difference continues to increase with more examples.

### 2. Logit Attribution

Here, I present the results for the 5-shot setting because the logit difference is largest in this setting and therefore may show the most robust patterns.

- `accumulated_resid`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/accumulated_residual_logit_diff.html"
    title="Accumulated residual logit difference"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

Hmm... The logit difference begins to increase from layer 13. However, it is still hard to tell which layer contributes most strongly to the logit difference. To examine this more directly, I next look at `decompose_resid`.

- `decompose_resid`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/decomposed_residual_logit_diff.html"
    title="Decomposed residual logit difference"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

Here, we can see that 27_attn_out has the largest logit difference. Overall, the attention outputs contribute more strongly than the MLP outputs. This suggests that the main prediction-shaping signal is more concentrated in attention outputs than in MLP outputs, motivating a head-level analysis.

- `stack_head_results`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/per_head_logit_diff.html"
    title="Per-head logit difference"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

L27H02 appears to be the most important head. This may explain why 27_attn_out showed the largest logit difference in the decomposed residual analysis. Next, I examine whether this head is sensitive to changes in the query.

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

L27H02 shows a large paired flip logit difference, suggesting that this head is sensitive to the query. Although the paired static logit difference may also look large at first glance, its scale is much smaller than that of the flip logit difference.

### 3. Activation Patching

In this section, I present the results for the 2-shot setting. This setting is a reasonable choice because the computational cost is moderate while the model already achieves perfect accuracy. I start with `resid_pre` patching.

- `resid_pre`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/2-shot/act_patch_resid_pre.html"
    title="2-shot resid_pre activation patching"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

Thanks to the controlled prompt format and dataset design, the patching results are easy to interpret. Position 19 corresponds to the query input position, that is, the input word of the query whose mapping should be executed. Position 22 corresponds to the final token position.

In the previous section, attention heads in layer 27 showed the largest logit differences. However, the resid_pre patching results suggest that query information is first localized at the query input position and later appears at the final token position. This suggests that some heads may act as query-information movers, transferring information from the query token into the residual stream at the final token.

If such heads write query information into the final token residual stream, then it becomes plausible to search for execution-head candidates among the heads operating at the final token position. Therefore, I next examine attention-head output patching at the final token.

- `attn_head_out_last_pos`

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/2-shot/act_patch_attn_head_out_last_pos.html"
    title="2-shot attn_head_out_last_pos activation patching"
    style="width: 100%; max-width: 1000px; height: 620px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

Even in the 2-shot setting, L27H02 remains the most important head in the final-token attention-head output patching experiment. This is consistent with the logit attribution result from the 5-shot setting, suggesting that L27H02 is not only strongly correlated with the correct logit direction but also causally important for recovering the clean prediction.

### 4. Validation on Held-Out Data

To test whether the identified heads are important by chance, I validate them on held-out data. I select 10 heads from each group based on the patching score: the top 10 heads, the bottom 10 heads, and 10 random heads. The table below shows the average patching effect for each group.

| n-shot | Positive mean `average_metric` | Random mean `average_metric` | Negative mean `average_metric` | 
|---:|---:|---:|---:|
| 0 | 0.035395 | -0.000947 | -0.021785 |
| 1 | 0.041080 | -0.000389 | -0.017664 |
| 2 | 0.039074 | 0.000275 | -0.014328 |
| 3 | 0.039596 | 0.001791 | -0.012629 |
| 4 | 0.038977 | 0.000756 | -0.013414 |
| 5 | 0.038947 | 0.000771 | -0.013131 |

Across all n-shot settings, the positive heads consistently show a positive average patching effect, while the random heads stay close to zero and the negative heads show negative effects. This suggests that the identified positive and negative heads are not artifacts of the dataset, but capture reproducible causal effects on held-out prompts.

In particular, L27H02 consistently shows the largest effect across all n-shot settings.

| n-shot | Positive validation | Random validation | Negative validation |
|---:|---|---|---|
| 0 | L27H02 (`0.080117`) | L22H04 (`0.011895`) | L27H07 (`-0.010273`) |
| 1 | L27H02 (`0.119355`) | L22H04 (`0.011914`) | L19H02 (`-0.011602`) |
| 2 | L27H02 (`0.112461`) | L22H04 (`0.013008`) | L27H17 (`-0.006738`) |
| 3 | L27H02 (`0.111094`) | L22H04 (`0.014687`) | L27H17 (`-0.004414`) |
| 4 | L27H02 (`0.111348`) | L22H04 (`0.013887`) | L27H17 (`-0.004395`) |
| 5 | L27H02 (`0.111973`) | L22H04 (`0.013555`) | L27H17 (`-0.004141`) |


## Discussion

In this post, I aimed to investigate the conditions under which a task vector is executed by searching for attention heads that implement a mapping of the form: task vector + query input → output. The basic assumption is that if a component has a large causal effect under activation patching, then it may contain information that is important for producing the query-dependent answer. Under this view, I identified L27H02 as the head with the largest contribution across several analyses.

However, the current results only show that this head is important for recovering the correct output. They do not yet explain what information the head receives, what information it writes to the residual stream, or how this information is routed through the model. To answer these questions, a more detailed circuit analysis is needed. One possible direction is to follow the [IOI paper](https://arxiv.org/pdf/2211.00593) and examine the representations read and written by the head using projections onto relevant unembedding directions. Another direction is to use path patching to trace how information flows from earlier components to this head and then to the final prediction.

If this analysis reveals how query information and task-context information are combined and routed, it may shed light on what information [Function Vectors](https://arxiv.org/pdf/2310.15213) or [Task Vectors](https://arxiv.org/pdf/2310.15916) actually contain. In particular, it may help explain why injecting such vectors can sometimes recover task behavior even in zero-shot inference, and under what conditions such interventions succeed or fail.

## Additional Notes

In the 2-shot setting, the average attention patterns of the top 10 heads with the largest patching effects are shown below. 

<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
  <iframe
    src="/assets/html/execution-head/2-shot/top_positive_attention.html"
    title="2-shot top positive attention heads"
    style="width: 100%; max-width: 1000px; height: 760px; border: 0;"
    loading="lazy"
  ></iframe>
</div>

Interestingly, L27H02 strongly attends to its own position. This suggests a possible hypothesis: L27H02 may read information already present in the residual stream at the final token, rather than directly copying information from another token position. I leave a more detailed investigation of these heads for future work.

## References

- Roee Hendel, Mor Geva, and Amir Globerson. [In-Context Learning Creates Task Vectors](https://arxiv.org/pdf/2310.15916). arXiv:2310.15916, 2023.
- Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. [Function Vectors in Large Language Models](https://arxiv.org/pdf/2310.15213). ICLR 2024.
- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/pdf/2211.00593). arXiv:2211.00593, 2022.
