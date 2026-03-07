---
layout: post
title: Reproduction of Refusal Direction in LLMs
date: 2026-03-07
description: Notes on reproducing "Refusal in Language Models Is Mediated by a Single Direction" with causal intervention on the residual stream.
tags: mechanistic-interpretability causal-intervention
categories: exploratory
thumbnail: /assets/img/gemma-2b-itrefusal_fig3_reproduce.png
---

## Motivation

Large language models are trained to refuse harmful requests through post-training techniques such as safety fine-tuning. However, these safeguards are not always reliable, and models may still produce harmful responses in certain situations.

Meanwhile, recent progress in mechanistic interpretability suggests that some concepts can be represented as a single direction in the residual stream of a transformer model.

If this is the case, it raises an interesting possibility: rather than relying solely on post-training alignment, we might be able to induce or suppress refusal behaviors by directly manipulating the model's internal representations.

_Refusal in Language Models Is Mediated by a Single Direction_ explores this idea. In this post, I reproduce its experiment to see how the method works in practice.

## Core Idea

[Summarize the main hypothesis of the paper]

Conceptually:

```
[core idea / representation hypothesis]
```

[Explain what this idea implies]

## Experimental Setup

[Briefly explain how the experiment works]

### 1. [Step name]

[Explain what happens in this step]

### 2. [Step name]

```
[equation or representation definition]
```

[Explanation]

### 3. [Step name]

- [metric or criterion]
- [metric or criterion]
- [metric or criterion]

[Explanation]

### 4. [Step name]

[Explain intervention]

```
[operation or equation]
```

```
[operation or equation]
```

[Explanation]

## Implementation

[Briefly describe your setup]

- [library]
- [model]
- [environment]

[Explain what you implemented]

Code:  
[GitHub link]

## Result

[Describe what you observed]

<img src="/assets/img/gemma-2b-itrefusal_fig3_reproduce.png" alt="Result: refusal scores for harmless (induced) and harmful (bypassed) prompts" style="max-width: 100%; width: 560px;">

[Explain the result]

## Interpretation

[What the result suggests]

[Connection to the paper's hypothesis]

## Reflection

[Your thoughts about the experiment]

[What you found interesting]

[Possible future questions]

I reproduced the main experiment from the paper **"Refusal in Language Models Is Mediated by a Single Direction"** ([arXiv:2406.11717](https://arxiv.org/pdf/2406.11717)). This post summarizes the setup and findings.

**Repository:** [JakeFRCSE/refusal-direction-reproduction](https://github.com/JakeFRCSE/refusal-direction-reproduction)
