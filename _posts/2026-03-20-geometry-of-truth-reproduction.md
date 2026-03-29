---
layout: post
title: Reproduction of Geometry of Truth
date: 2026-03-20
description: Notes on reproducing "Geometry of Truth" with causal intervention and visualization.
tags: mechanistic-interpretability linear-structure
categories: exploratory
thumbnail: /assets/img/geometry-of-truth/layer_12_Llama-2-13b-hf.png
---

## TL;DR

I reproduced the main experiments from **"Geometry of Truth in Language Models"** ([arXiv:2310.06824](https://arxiv.org/pdf/2310.06824)), including localization, PCA visualization, generalization, and NIE-based intervention analysis. In my reproduction, Llama-2-13b shows the same overall pattern as the paper: a clearer linear truth structure in PCA, better cross-dataset generalization when probes are trained on two datasets, and substantially stronger intervention effects. In contrast, Llama-2-7b and smaller models do not show the same clean structure; instead, PCA suggests a more discrete clustered structure, and the weak NIE values suggest that the truth-related representation is not yet causally organized in the same way. The implemented repository is available **[here](https://github.com/JakeFRCSE/geometry-of-truth-reproduction)**.

## Motivation

In the reproduction of refusal direction in LLMs, the concept of intervention was restricted to refusal behavior. If the linear structure truly exists, it should be applicable to other concepts as well.

## Core Idea

The paper investigates the linear structure of the truth feature by applying PCA to the activations extracted from localized points.

## Experimental Setup

The experiments are conducted on various models.

The dataset is imported directly from the **[codebase](https://github.com/saprmarks/geometry-of-truth/tree/main/datasets)** of the paper.

### 1. Localizing the activation of truth feature

To localize causally implicated hidden states, the paper runs patching experiments on cached residual stream activations. Starting from a prompt `pF` whose final statement is false, it constructs a corresponding prompt `pT` by making the same statement true and caches residual stream activations at every token position `i` and layer `l` for `pT`.

Then, for each `(i, l)`, it reruns the model on `pF` while swapping in the cached residual stream activation from `pT` (and letting this affect downstream computation). For each intervention, it measures causal influence by the difference in log probability between the tokens "TRUE" and "FALSE," and collects the most causally influential activations into groups.

### 2. Computing PCA of truth feature

Guided by the localized "truth" hidden states, the paper visualizes representation geometry with PCA. For each dataset, it takes the most downstream residual stream activation from the truth group (for example, LLaMA-2-13B uses the layer-15 residual stream over the end-of-sentence punctuation token). It also centers activations by subtracting the mean, and then applies PCA to obtain the leading principal components.

On curated true/false datasets with little variation in non-truth factors, the top principal components reveal clear linear structure: true and false statements separate in the projection.

### 3. Probing and generalization experiments

For the generalization experiments, the paper trains linear probes on one dataset and evaluates them on other datasets with different topics or surface forms. In addition to logistic regression (LR), the paper introduces mass-mean (MM) probing, which uses a difference-in-means direction together with a correction term intended to mitigate interference from features that are not orthogonal to the truth feature.

The main goal is to test whether the learned direction captures a more general notion of truth rather than features tied to one template. The paper also studies whether training on a dataset together with its "opposite" dataset, such as `cities + neg_cities` or `larger_than + smaller_than`, improves transfer to other datasets.

### 4. Measuring NIE for intervention

To test whether the learned probe directions are causally implicated in model outputs, the paper performs intervention experiments on the localized truth-related hidden states. For false-to-true and true-to-false settings, it adds or subtracts the probe direction at the selected token positions and layers, then measures how much the model's output moves toward labeling the statement as TRUE or FALSE.

The paper summarizes this causal effect with the normalized indirect effect (NIE). Intuitively, an NIE near 0 means that the intervention had little effect on the model's prediction, while an NIE near 1 means that the intervention shifted the model's output as strongly as moving from a genuinely false statement to a genuinely true one, or vice versa.

Using this metric, the paper argues that MM identifies directions that are more causally implicated in model outputs than LR, even when its classification accuracy is slightly lower than LR.

## Implementation

This implementation relies on the `TransformerLens` and `nnsight` libraries for convenient activation caching and interventions. The experiments are conducted on various models. The list of models is shown below:

| Family     | Model                       |
| ---------- | --------------------------- |
| Meta Llama | `meta-llama/Llama-2-7b-hf`  |
| Meta Llama | `meta-llama/Llama-2-13b-hf` |
| Meta Llama | `meta-llama/Llama-3.1-8B`   |
| Meta Llama | `meta-llama/Llama-3.2-1B`   |
| Meta Llama | `meta-llama/Llama-3.2-3B`   |
| Pythia     | `EleutherAI/pythia-160m`    |
| Pythia     | `EleutherAI/pythia-410m`    |
| Pythia     | `EleutherAI/pythia-1b`      |
| Pythia     | `EleutherAI/pythia-1.4b`    |
| Pythia     | `EleutherAI/pythia-2.8b`    |
| Pythia     | `EleutherAI/pythia-6.9b`    |
| Qwen       | `Qwen/Qwen1.5-1.8B`         |
| Gemma      | `google/gemma-2b-it`        |

## Result

### Paper Results

<div style="display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap;">
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>Localization of truth feature (Figure 1 in the paper)</strong>
    <img
      src="/assets/img/geometry-of-truth/Patching_paper.png"
      alt="Result: localization of truth feature (Llama-2-13b)"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem;">The paper's result for Llama-2-13b.</p>
  </div>
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>PCA of truth feature (Figure 2 in the paper)</strong>
    <img
      src="/assets/img/geometry-of-truth/PCA_paper.png"
      alt="Result: PCA of truth feature (Llama-2-70b)"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem;">The paper's result for Llama-2-70b.</p>
  </div>
</div>

### Reproduced Result

The reproduced results are shown below:

<div style="display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap;">
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>Localization of truth feature (Llama-2-13b)</strong>
    <img
      src="/assets/img/geometry-of-truth/cities_intervention_heatmap_Llama-2-13b-hf.png"
      alt="Result: localization of truth feature"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem;">The truth feature is localized at layer 12 of Llama-2-13b, as in the paper.</p>
  </div>
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>PCA of truth feature (Llama-2-13b)</strong>
    <img
      src="/assets/img/geometry-of-truth/layer_12_Llama-2-13b-hf.png"
      alt="Result: PCA of truth feature"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem;">The PCA of the truth feature shows a consistent linear structure, as in the paper.</p>
  </div>
</div>

### Extended Results

The experiment is extended to the Llama-2-7b model to test whether the linear structure exists in the smaller model.

<div style="display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap;">
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>PCA of truth feature (Llama-2-7b)</strong>
    <img
      src="/assets/img/geometry-of-truth/layer_12_Llama-2-7b-hf.png"
      alt="Result: PCA of truth feature"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem;">The truth feature is localized at layer 12 of Llama-2-7b, but the linear structure does not form.</p>
  </div>
  <div style="flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column;">
    <strong>PCA of truth feature (Pythia-160m)</strong>
    <img
      src="/assets/img/geometry-of-truth/layer_05_pythia_160m.png"
      alt="Result: PCA of truth feature"
      style="max-width: 100%; width: 100%; height: auto;"
    >
    <p style="margin-top: auto; padding-top: 0.5rem;">Pythia-160m's PCA shows a similar structure to Llama-2-7b.</p>
  </div>
</div>

The results for other models are available **[here](https://drive.google.com/drive/folders/1Azb5cNOOTnu5KtHXSZw9waYHPEoySMOT)**.

### Generalization

The generalization experiments were evaluated at layer 10 for Llama-2-7b and at layer 15 for Llama-2-13b. The reproduced figures are shown side by side with the corresponding figures from the paper.

**Llama-2-7b**

<div style="display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap;">
  <div style="flex: 1 1 320px; min-width: 280px; display: grid; grid-template-rows: auto 260px auto;">
    <strong>Paper</strong>
    <div style="display: flex; align-items: center; justify-content: center; padding: 0.75rem 0;">
      <img
        src="/assets/img/geometry-of-truth/generalization-llama-2-7b.png"
        alt="Generalization result from the paper for Llama-2-7b"
        style="max-width: 100%; max-height: 100%; width: auto; height: auto;"
      >
    </div>
    <p style="margin-top: 0; padding-top: 0.75rem; border-top: 1px solid rgba(0, 0, 0, 0.15);">Generalization result reported in the paper.</p>
  </div>
  <div style="flex: 1 1 320px; min-width: 280px; display: grid; grid-template-rows: auto 260px auto;">
    <strong>Reproduced</strong>
    <div style="display: flex; align-items: center; justify-content: center; padding: 0.75rem 0;">
      <img
        src="/assets/img/geometry-of-truth/generalization-llama-2-7b-layer10.png"
        alt="Reproduced generalization result for Llama-2-7b at layer 10"
        style="max-width: 100%; max-height: 100%; width: auto; height: auto;"
      >
    </div>
    <p style="margin-top: 0; padding-top: 0.75rem; border-top: 1px solid rgba(0, 0, 0, 0.15);">Reproduced result at layer 10.</p>
  </div>
</div>

**Llama-2-13b**

<div style="display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap;">
  <div style="flex: 1 1 320px; min-width: 280px; display: grid; grid-template-rows: auto 260px auto;">
    <strong>Paper</strong>
    <div style="display: flex; align-items: center; justify-content: center; padding: 0.75rem 0;">
      <img
        src="/assets/img/geometry-of-truth/generalization-llama-2-13b.png"
        alt="Generalization result from the paper for Llama-2-13b"
        style="max-width: 100%; max-height: 100%; width: auto; height: auto;"
      >
    </div>
    <p style="margin-top: 0; padding-top: 0.75rem; border-top: 1px solid rgba(0, 0, 0, 0.15);">Generalization result reported in the paper.</p>
  </div>
  <div style="flex: 1 1 320px; min-width: 280px; display: grid; grid-template-rows: auto 260px auto;">
    <strong>Reproduced</strong>
    <div style="display: flex; align-items: center; justify-content: center; padding: 0.75rem 0;">
      <img
        src="/assets/img/geometry-of-truth/generalization-llama-2-13b-rep-layer15.png"
        alt="Reproduced generalization result for Llama-2-13b at layer 15"
        style="max-width: 100%; max-height: 100%; width: auto; height: auto;"
      >
    </div>
    <p style="margin-top: 0; padding-top: 0.75rem; border-top: 1px solid rgba(0, 0, 0, 0.15);">Reproduced result at layer 15.</p>
  </div>
</div>

The exact numbers are not perfectly reproduced. However, when the probe is trained on two datasets, probing performance improves in both models, so I think this still reproduces the paper's main claim about generalization.

### NIE

NIE was measured on Llama-2-7b by intervening on tokens from layers 5 to 10 and on Llama-2-13b by intervening on tokens from layers 8 to 14. Across both models, MM generally produced stronger intervention results than LR, and the difference is especially clear on Llama-2-13b.

<table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
  <thead>
    <tr>
      <th style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: left;">Train set</th>
      <th style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: left;">Probe</th>
      <th style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">Llama-2-13b false-to-true</th>
      <th style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">Llama-2-13b true-to-false</th>
      <th style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">Llama-2-7b false-to-true</th>
      <th style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">Llama-2-7b true-to-false</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>cities</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">LR</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.138</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.100</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.064</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.005</td>
    </tr>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>cities</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">MM</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.663</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.770</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.014</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.032</td>
    </tr>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>cities_combined</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">LR</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.205</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.353</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.014</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.002</td>
    </tr>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>cities_combined</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">MM</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.697</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.811</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.017</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.015</td>
    </tr>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>larger_than</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">LR</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.197</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.169</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.081</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.012</td>
    </tr>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>larger_than</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">MM</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.491</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.600</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.071</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.003</td>
    </tr>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>larger_than_combined</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">LR</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.070</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.070</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.007</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.000</td>
    </tr>
    <tr>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;"><code>larger_than_combined</code></td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem;">MM</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.214</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.332</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">-0.007</td>
      <td style="border: 1px solid rgba(0, 0, 0, 0.2); padding: 0.5rem; text-align: right;">0.000</td>
    </tr>
  </tbody>
</table>

Note that intervention does not meaningfully work on the 7B model: the NIE values stay close to zero and are much smaller than those of the 13B model. This is consistent with the earlier observation that the linear truth structure does not clearly form in the smaller model.

## Interpretation

### What the result suggests

The reproduced results suggest that the linear structure of the truth feature clearly exists in the larger models; however, it does not form in the smaller models.

The intervention results make this interpretation more concrete. In PCA, Llama-2-13b shows a qualitatively clear separation between true and false statements, while Llama-2-7b and Pythia-160m do not show the same clean linear organization. This qualitative gap is reflected in the causal results: the 13B model shows large NIE values, whereas the 7B model stays near zero across datasets.

The generalization results point in the same direction. Even though the exact numbers are not perfectly reproduced, probes trained on two datasets generalize better than probes trained on a single dataset, which suggests that interference by features that are not orthogonal to the truth feature is mitigated.

### Follow-up Research Questions

1. How is the discrete clustered linear structure transformed into a continuous linear structure?
2. How are the linear structure and intervention related?
