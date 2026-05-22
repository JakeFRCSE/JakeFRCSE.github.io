---
layout: about
title: about
permalink: /
subtitle:

profile:
  align: left
  image: IMG_4836.JPG
  image_circular: true # crops the image to make it circular
  more_info: >
    <p>Gunsan, South Korea</p>

selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: true # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: true
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

Hi👋, this is Junhyeok! I recently graduated with a double major in Computer Science and French Language & Literature, and I am currently joining the **[Yonsei MLAI Lab](https://mlai.yonsei.ac.kr/home)** as a research intern. My research focuses on mechanistic interpretability and AI safety, particularly on understanding how internal representations emerge, organize, and causally influence the behavior of large language models.

My work so far has centered on reproduction projects and independent experiments that probe the relationship between representational geometry and model behavior. I have reproduced **[refusal direction ablation](/blog/2026/refusal-direction-reproduction/)** to verify that a single direction in the residual stream mediates refusal, examined **[linear truth structure](/blog/2026/geometry-of-truth-reproduction/)** across multiple model families and scales using PCA, probing, and causal intervention, and investigated **[query-conditioned execution heads](/blog/2026/execution-head/)** to identify attention heads that causally support task execution in in-context learning.

Currently, I am interested in how task representations and computational circuits emerge during in-context learning, especially through causal mechanisms such as execution heads and function vectors. More broadly, I am interested in representation geometry, causal intervention, and scalable interpretability methods for understanding and building reliable language models.
