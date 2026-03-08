---
layout: post
title: Reproduction of Refusal Direction in LLMs
date: 2026-03-07
description: Notes on reproducing "Refusal in Language Models Is Mediated by a Single Direction" with causal intervention on the residual stream.
tags: mechanistic-interpretability causal-intervention
categories: exploratory
thumbnail: /assets/img/gemma-2b-itrefusal_fig3_reproduce.png
---

## TL;DR

I reproduced the main experiment from the paper **"Refusal in Language Models Is Mediated by a Single Direction"** ([arXiv:2406.11717](https://arxiv.org/pdf/2406.11717)).

The paper proposes that refusal behavior corresponds to a single direction in the residual stream of a transformer model. By adding this direction, the model can be induced to refuse harmless prompts, while removing it suppresses refusal on harmful prompts.

Using **Gemma 2B**, I reproduced the main experiment and observed similar trends: injecting the refusal direction increases refusal on harmless prompts, while ablating it reduces refusal on harmful prompts.

## Motivation

Large language models are trained to refuse harmful requests through post-training techniques such as safety fine-tuning. However, these safeguards are not always reliable, and models may still produce harmful responses in certain situations.

Meanwhile, recent progress in mechanistic interpretability suggests that some concepts can be represented as a single direction in the residual stream of a transformer model.

If this is the case, it raises an interesting possibility: rather than relying solely on post-training alignment, we might be able to induce or suppress refusal behaviors by directly manipulating the model's internal representations.

_Refusal in Language Models Is Mediated by a Single Direction_ explores this idea. In this post, I reproduce its experiment to see how the method works in practice.

## Core Idea

The paper hypothesizes that refusal behavior is represented by a single direction in the residual stream. Adding this direction induces refusal on harmless prompts, while removing it suppresses refusal on harmful prompts.

## Experimental Setup

The experiment is conducted on the **Gemma 2B** model.
The dataset is directly imported from the [codebase](https://github.com/andyrdt/refusal_direction/tree/main/dataset/splits) of the paper.

### 1. Extract residual stream representations from harmful and harmless prompts

Residual stream representations are extracted from harmful and harmless prompts.
The prompts are constructed by inserting harmful or harmless instructions into the chosen model's instruction format.
After tokenization, the prompts are passed to the model to obtain the residual stream activations.
These representations are collected from all layers of the model at post-instruction token positions (Appendix C.3).

### 2. Compute candidate refusal directions from the difference between mean representations

Candidate refusal directions are computed from the difference between mean representations.
The representations at the same layer and position index are averaged over the batch dimension.
The difference between the averaged representations is taken to compute the candidate refusal directions.
Thus, the final candidate refusal directions have shape (num_layers, num_post_instruction_tokens, hidden_size).

### 3. Select the optimal direction using bypass/induce scores

The optimal direction is selected following the selection algorithm described in the paper (Appendix C).
However, evaluating the scores via generation is computationally expensive.
Therefore, the proxy metric proposed in the paper is used to reduce the cost (Appendix B).

## Implementation

The implementation is available [here](https://github.com/JakeFRCSE/refusal-direction-reproduction).
The implementation relies on the `TransformerLense` library for convenient activation caching and interventions.
Due to limited computational resources, the experiments are conducted only on google/gemma-2b-it.

## Result

### Paper Results

**Ablating the refusal direction (Figure 1 in the paper)**

<img src="/assets/img/fig1_refusal.png" alt="Result: refusal scores of baseline and ablated runs" style="max-width: 100%; width: 560px;">

As shown in the paper, ablating the refusal direction reduces refusal behavior on harmful prompts.

**Adding the refusal direction (Figure 3 in the paper)**

<img src="/assets/img/fig3_refusal.png" alt="Result: refusal scores of baseline and induced runs" style="max-width: 100%; width: 560px;">

Adding the refusal direction induces refusal behavior even for harmless prompts.

### Reproduced Result

Our reproduction produces the following result:

<img src="/assets/img/gemma-2b-itrefusal_fig3_reproduce.png" alt="Result: refusal scores for harmless (induced) and harmful (bypassed) prompts" style="max-width: 100%; width: 560px;">

The reproduced results show similar trends: adding the refusal direction increases refusal on harmless prompts, while removing the direction reduces refusal on harmful prompts.

## Interpretation

### What the result suggests

The reproduced results show that manipulating a single direction in the residual stream systematically changes the model’s refusal behavior.
Adding the direction increases refusal even on harmless prompts, while removing the direction reduces refusal on harmful prompts.

This suggests that the model’s refusal behavior is strongly associated with a specific representation in the residual stream.
Intervening on this representation directly alters the model’s decision to refuse or comply with a prompt.

### Connection to the paper's hypothesis

These findings are consistent with the paper’s hypothesis that refusal behavior is mediated by a single direction in the residual stream.

If refusal were distributed across many unrelated representations, manipulating a single direction would not produce such consistent changes in behavior.
However, the experimental results show that adding or removing this direction reliably induces or suppresses refusal behavior.

## Reflection

Reproducing the results of Refusal in Language Models Is Mediated by a Single Direction provided a useful perspective on how safety behaviors can emerge from relatively simple internal representations. In particular, the experiment suggests that refusal behavior can be manipulated through a single direction in the residual stream, which supports the idea that certain high-level behaviors in language models may correspond to relatively localized representations.

One interesting aspect of the implementation is that the intervention is applied across all layers and token positions when ablating the refusal direction. This design choice appears to ensure that the target direction is fully removed from the model's computation. However, this raises an additional question: it remains unclear at which layer the refusal signal is most strongly represented. A more detailed layer-wise analysis could provide further insight into where refusal behavior is actually formed in the network.

Several potential extensions remain for future exploration. The rank-one weight editing method proposed in the paper was not reproduced in this experiment due to time constraints. Investigating this approach could provide further insight into how internal representations can be modified through weight-level interventions.
