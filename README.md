# Generative-Model-Paper

The following papers are my research interests (some papers I have recently read🥸).

---

## Multimodal：
+ 2021.02.26 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP)

## Language models：
+ 2023.03.15 [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) (GPT4)
+ 2022.01.26 [InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://openai.com/research/instruction-following)
+ 2020.05.28 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT3)
+ 2019.02.14 [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) (GPT2)
+ 2018.06.11 [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised) (GPT)

## Large Language Diffusion model:
+ 2025.02.14 [Large Language Diffusion Models](https://arxiv.org/pdf/2502.09992)
+ 2024.10.24 [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)
+ 2024.10.23 [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891) (Adaptation)
+ 2024.06.06 [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736)
+ 2023.10.25 [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) (SEDD)
+ 2023.03.12 [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574) (Survey)
+ 2022.11.30 [Score-based Continuous-time Discrete Diffusion Models](https://arxiv.org/abs/2211.16750) (Score-based)
+ 2022.11.02 [Concrete Score Matching: Generalized Score Matching for Discrete Data](https://arxiv.org/abs/2211.00802)
  + The first to propose concrete score for modeling discrete data.
+ 2022.05.30 [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987) (CTMC)
  + First complete continuous time framework for discrete diffusion.
+ 2021.07.07 [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) (D3PM)
  + Diffusion models with discrete state spaces as a competitive model class for large scale text or image generation, however, it trains and samples the model in discrete time.

## ImageNet Generation SOTA:
+ Ranking list: https://paperswithcode.com/sota/image-generation-on-imagenet-256x256
+ 2025.02.17 [Diffusion Models without Classifier-free Guidance](https://arxiv.org/abs/2502.12154v1) (FID:1.34)

## Diffusion model:
+ 2022.07.26 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (classifier free guidance)
+ 2021.05.11 [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (classifier guidance)
+ 2020.11.26 [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
  + Unified framework for 19's score-based method and 20's DDPM.
+ 2020.10.06 [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM)
+ 2020.06.19 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM)
+ 2019.07.12 [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) (Score-based method)
+ 2015.03.12 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

## Diffusion model for customization:
+ 2024.07.08 [Ada-adapter:Fast Few-shot Style Personlization of Diffusion Model with Pre-trained Image Encoder](https://arxiv.org/abs/2407.05552) (Ada-adapter)
+ 2023.08.13 [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721) (IP-Adapter)
+ 2023.02.10 [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (ControlNet)
+ 2022.08.25 [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) (DreamBooth)
+ 2022.08.02 [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618) (Textual Inversion)

## Trustworthy AI

## Efficient AI:
+ 2023.12.06 [On the Diversity and Realism of Distilled Dataset: An Efficient Dataset Distillation Paradigm](https://arxiv.org/abs/2312.03526) (RDED)
+ 2023.06.22 [Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective](https://arxiv.org/abs/2306.13092) (SRe2L)
+ 2022.07.20 [DC-BENCH: Dataset Condensation Benchmark](https://arxiv.org/abs/2207.09639)
+ 2022.06.29 [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486) (scaling law)
+ 2022.03.22 [Dataset Distillation by Matching Training Trajectories](https://arxiv.org/abs/2203.11932) (TM: matching trajectory)
+ 2022.03.03 [CAFE: Learning to Condense Dataset by Aligning Features](https://arxiv.org/abs/2203.01531) (matching feature)
+ 2021.10.08 [Dataset Condensation with Distribution Matching](https://arxiv.org/abs/2110.04181) (DM: matching distribution)
+ 2020.06.10 [Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929) (DC: matching gradient)

## Self-Supervised learning:
+ 2020.06.13 [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733) (BYOL)

## Architecture:
Transformer/...

## Classic:
AlexNet/GAN...

---

## Code
[TinyLlama](https://github.com/jzhang38/TinyLlama)

## Blog
[Language Modeling by Estimating the Ratios of the Data Distribution](https://aaronlou.com/blog/2024/discrete-diffusion/) (SEDD)

## Video
+ Score Matching [Youtube](https://www.youtube.com/watch?v=B4oHJpEJBAA) [Bilibili](https://www.bilibili.com/video/BV1h8FZeBEqj?vd_source=5519a229401f5e71a4a2b1c367c2a569)
+ Diffusion and Scored-based Generative Model (by Yang Song) [Youtube](https://www.youtube.com/watch?v=wMmqCMwuM2Q) [Bilibili](https://www.bilibili.com/video/BV1LpNweeE3q?vd_source=5519a229401f5e71a4a2b1c367c2a569)
