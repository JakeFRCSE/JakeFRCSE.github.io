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
    <p>Seoul, South Korea</p>

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

Hi👋, this is Junhyeok! I graduated with a double major in Computer Science and French Language & Literature, and I am currently working at **[MLAI Lab@Yonsei](https://mlai.yonsei.ac.kr/home)** as a research intern. My research focuses on mechanistic interpretability and AI safety.

Before joining the lab, my work centered on reproductions and independent experiments probing how representational geometry relates to model behavior. I reproduced **[refusal direction ablation](/blog/2026/refusal-direction-reproduction/)** to verify that a single direction in the residual stream mediates refusal, examined **[linear truth structure](/blog/2026/geometry-of-truth-reproduction/)** across model families and scales using PCA, probing, and causal intervention, and investigated **[query-conditioned execution heads](/blog/2026/execution-head/)** that causally support task execution in in-context learning. Each of these reads an internal state through a probe or a direction fixed in advance, for a behavior chosen in advance.

Currently, I am interested in methods that translate the internal activations of LLMs into natural language descriptions, such as natural language autoencoders, activation oracles, and LatentQA-style decoding. I am particularly interested in improving the faithfulness of these explanations. If you are interested in these topics, feel free to reach out!
