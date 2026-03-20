---
layout: post
title: Reproduction of Geometry of Truth
date: 2026-03-20
description: Notes on reproducing "Geometry of Truth" with causal intervention and visualization.
tags: mechanistic-interpretability causal-intervention
categories: exploratory
thumbnail: /assets/img/geometry-of-truth/layer_12_Llama-2-13b-hf.png
---

## TL;DR

I reproduced the experiments from the paper **"Geometry of Truth in Language Models"** ([arXiv:2310.06824](https://arxiv.org/pdf/2310.06824)).

The paper localizes the activation of the truth feature and visualizes the linear structure of the truth feature.

While reproducing the experiments, I observed that the linear structure does not form in the Llama-2-7b model, and the PCA results show a structure other than linear.

The implemented repository is available **[here](https://github.com/JakeFRCSE/geometry-of-truth-reproduction)**.

## Motivation

In the reproduction of refusal direction in LLMs, the concept of intervention was restricted to refusal behavior.

If the linear structure truly exists, it should be applicable to other concepts as well.

## Core Idea

The paper investigates the linear structure of the truth feature by applying PCA to the activations extracted from localized points.

## Experimental Setup

The experiments are conducted on various models.
The dataset is imported directly from the [codebase](https://github.com/saprmarks/geometry-of-truth/tree/main/datasets) of the paper.

### 1. Localize activation of truth feature

To localize causally implicated hidden states, the paper runs patching experiments on cached residual stream activations. Starting from a prompt `pF` whose final statement is false, it constructs a corresponding prompt `pT` by making the same statement true and caches residual stream activations at every token position `i` and layer `l` for `pT`.

Then, for each `(i, l)`, it reruns the model on `pF` while swapping in the cached residual stream activation from `pT` (and letting this affect downstream computation). For each intervention, it measures causal influence by the difference in log probability between the tokens "TRUE" and "FALSE," and collects the most causally influential activations into groups.

### 2. Compute PCA of truth feature

Guided by the localized "truth" hidden states, the paper visualizes representation geometry with PCA. For each dataset, it takes the most downstream residual stream activation from the truth group (for example, LLaMA-2-13B uses the layer-15 residual stream over the end-of-sentence punctuation token). It also centers activations by subtracting the mean, and then applies PCA to obtain the leading principal components.

On curated true/false datasets with little variation in non-truth factors, the top principal components reveal clear linear structure: true and false statements separate in the projection.

## Implementation

The implementation is available **[here](https://github.com/JakeFRCSE/geometry-of-truth-reproduction)**.
This implementation relies on the `TransformerLens` and `nnsight` libraries for convenient activation caching and interventions.

## Result

### Paper Results

**Localization of truth feature (Figure 1 in the paper)**

<img src="/assets/img/geometry-of-truth/Patching_paper.png" alt="Result: localization of truth feature (Llama-2-13b)" style="max-width: 100%; width: 560px;">

The paper's result for Llama-2-13b.

**PCA of truth feature (Figure 2 in the paper)**

<img src="/assets/img/geometry-of-truth/PCA_paper.png" alt="Result: PCA of truth feature (Llama-2-70b)" style="max-width: 100%; width: 560px;">

The paper's result for Llama-2-70b.

### Reproduced Result

The reproduced results are shown below:

**Localization of truth feature (Llama-2-13b)**

<img src="/assets/img/geometry-of-truth/layer_12_Llama-2-13b-hf.png" alt="Result: localization of truth feature" style="max-width: 100%; width: 560px;">

The truth feature is localized at layer 12 of Llama-2-13b, as in the paper.

**PCA of truth feature (Llama-2-13b)**

<img src="/assets/img/geometry-of-truth/pca_Llama-2-13b-hf.png" alt="Result: PCA of truth feature" style="max-width: 100%; width: 560px;">

The PCA of the truth feature shows a consistent linear structure, as in the paper.

Results for other models are available **[here](https://drive.google.com/drive/folders/1Azb5cNOOTnu5KtHXSZw9waYHPEoySMOT)**.

### Extended Results

The experiment is extended to the Llama-2-7b model to test whether the linear structure exists in the smaller model.

**PCA of truth feature (Llama-2-7b)**

<img src="/assets/img/geometry-of-truth/layer_12_Llama-2-7b-hf.png" alt="Result: PCA of truth feature" style="max-width: 100%; width: 560px;">

The truth feature is localized at layer 12 of Llama-2-7b, but the linear structure does not form.

Pythia-160m's PCA shows a similar structure to Llama-2-7b.

**PCA of truth feature (Pythia-160m)**

<img src="/assets/img/geometry-of-truth/layer_05_pythia_160m.png" alt="Result: PCA of truth feature" style="max-width: 100%; width: 560px;">

## Interpretation

### What the result suggests

The reproduced results suggest that the linear structure of the truth feature clearly exists in the larger models; however, it does not form in the smaller models that show a similar structure.

### Follow-up Research Questions

1. How is the "failed" linear structure formed?
2. How are the linear structure and intervention related?
3. When does the linear structure form?
