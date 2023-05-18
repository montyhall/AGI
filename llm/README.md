# Large Language Models

<!-- ![Alt Text](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/survey-gif-test.gif) -->
<img src="https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/survey-gif-test.gif" width="900" height="500" />

Some other overviews:
* [The Practical Guides for Large Language Models](https://github.com/Mooler0410/LLMsPracticalGuide)

## Data

* [P3 (Public Pool of Prompts)](https://huggingface.co/datasets/bigscience/P3) `a collection of prompted English datasets covering a diverse set of NLP tasks`
* [chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications#fn1trot6ka6e2)
* [Natural Instructions](https://github.com/allenai/natural-instructions) Paper: [SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/pdf/2204.07705.pdf)
* [Raft](https://huggingface.co/datasets/ought/raft) `Real-world Annotated Few-shot Tasks (RAFT) dataset is an aggregation of English-language datasets found in the real world`
* [The Pile](https://pile.eleuther.ai/) `..a 825 GiB diverse, open source language modelling data set that consists of 22 smaller, high-quality datasets combined together.` [paper](https://arxiv.org/abs/2101.00027)
* [The Stack](https://www.bigcode-project.org/docs/about/the-stack/) `...a 6.4 TB dataset of permissively licensed source code in 358 programming languages`

## LLM models

See [list](https://en.wikipedia.org/wiki/Large_language_model) on Wikipedia. Another [here](https://github.com/Hannibal046/Awesome-LLM)
* [Bloom](https://huggingface.co/bigscience/bloom)
  - [Bloom-LORA](https://github.com/linhduongtuan/BLOOM-LORA)
  - [Petals](https://github.com/bigscience-workshop/petals)
* [NVIDIA's Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) `adding programmable guardrails to LLM`
* [H2O GPT](https://github.com/h2oai/h2ogpt) and [HF](https://huggingface.co/h2oai) [![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

* [Llama](https://github.com/facebookresearch/llama) Fill this [Form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) to get access to the weights. [A brief history of LLaMA models](https://agi-sphere.com/llama-models/)
* [lit-llama](https://github.com/Lightning-AI/lit-llama) Independent implementation of LLaMA that is fully open source under the [![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

* [MosaicML's MPT](https://github.com/mosaicml/llm-foundry) [![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
 [blog](https://www.mosaicml.com/blog/mpt-7b)
* [OpenLLaMA: An Open Reproduction of LLaMA](https://github.com/openlm-research/open_llama)
* [Redpajama](https://www.together.xyz/blog/redpajama) and [models](https://huggingface.co/togethercomputer)
* [Stable LLM](https://github.com/stability-AI/stableLM/) (cc license) (see [blog](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models))

See [here](https://github.com/eugeneyan/open-llms) for commercial license LLM models

## Fine Tuning

[Hyung Won Chung](https://scholar.google.com/citations?user=1CAlXvYAAAAJ&hl=en) in a talk recently at NYU nicely distinguished between fine tuning using MLE (what most people refer to as fine tuning -- a supervised learning task, or behavioral cloning in the language of RL) from learning the learning objective (RLHF). 

### Supervised Learning

[Good overview](https://lightning.ai/pages/community/article/understanding-llama-adapters/) of Fine tuning strategies by pytorch lightening. For a more research overview see paper: [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)

* [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* [gpt4all](https://github.com/nomic-ai/gpt4all)
* [PEFT](https://github.com/huggingface/peft)
* [koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
* [LLaMa-Adapter](https://github.com/zrrskywalker/llama-adapter)
* [OpenChatKit](https://github.com/togethercomputer/OpenChatKit)
* [vicuna](https://vicuna.lmsys.org/)
* pytorch tutorials:
  - [Understanding PEFT of LLMs: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
  - [PEFT with Lora](https://lightning.ai/pages/community/tutorial/lora-llm/)
  - [Finetuning LLMs on a Single GPU Using Gradient Accumulation](https://lightning.ai/pages/blog/gradient-accumulation/)
* [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)

### Learning the objective function
* [stackllama](https://huggingface.co/blog/stackllama)

# Compositional (Agents)
* [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
* [Chameleon](https://github.com/lupantech/chameleon-llm)
* [HuggingGPT](https://github.com/microsoft/JARVIS)
* [HuggingFace Transformers Agent](https://huggingface.co/docs/transformers/transformers_agents)

## Tools
* [dalai](https://github.com/cocktailpeanut/dalai)
* [FUTUREPEDIA](https://www.futurepedia.io/)AI Tools dir updated daily
* [langchain](https://python.langchain.com/en/latest/)
* [llamaindex](https://gpt-index.readthedocs.io/en/latest/index.html)
* [Semantic Kernel](https://github.com/microsoft/semantic-kernel) [hello sk](https://devblogs.microsoft.com/semantic-kernel/hello-world/)

## Vector DBs
* [Chroma](https://www.trychroma.com/)
* [FAISS](https://github.com/facebookresearch/faiss) (were doing this way back)
* [Milvus](https://milvus.io/)
* [Pinecone](https://www.pinecone.io/) <g-emoji class="g-emoji" alias="boom" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f4a5.png">💥</g-emoji> 2023-05-01 [Pinecone drops $100M investment on $750M valuation](https://techcrunch.com/2023/04/27/pinecone-drops-100m-investment-on-750m-valuation-as-vector-database-demand-grows/?utm_source=tldrfounders)
* [Vectera](https://vectara.com/)
* [Weaviate](https://weaviate.io/)

## Backends
* [deepspeed](https://github.com/microsoft/DeepSpeed)
* [LocalAI](https://github.com/go-skynet/LocalAI)
* [ray](https://www.ray.io/)
* [MLC LLM](https://github.com/mlc-ai/mlc-llm)
* [modal](https://modal.com/)
* pytorch-lightening
  - [comparisons](https://sebastianraschka.com/blog/2023/pytorch-faster.html) (pytorch, mixed precision, static graphs, deepspeed, fabric)
  - [Train 1T+ Model parameters](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html)
* [skypilot](https://skypilot.readthedocs.io/en/latest/index.html)

## General Resources
* [reentry](https://rentry.org/localmodelslinks)
* [Distributed AI Research Institute](https://www.dair-institute.org/)

## Security
* [categories of attacks](https://simonwillison.net/2023/Apr/25/dual-llm-pattern/) by Simon Willson (2023)
* [The AI Attack Surface Map v1.0](https://danielmiessler.com/blog/the-ai-attack-surface-map-v1-0/) (2023)

## Companies
* [startups @builtwithgenai](https://airtable.com/shr6nfE9FOHp17IjG/tblL3ekHZfkm3p6YT)
* [cohere](https://cohere.com/)

## Democratizaters
* [baseten](https://www.baseten.co/about)
* [huggingface](https://huggingface.co/)
* [Databrick's Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) Dolly [models on HF](https://huggingface.co/databricks)
* [Eleuther.ai](https://www.eleuther.ai/)
* [Lamini AI](https://lamini.ai/)
* [Ought](https://ought.org/)
* [Replicate](https://replicate.com/)
* [Runwayml](https://runwayml.com/)
* [Stable Diffusion Training with MosaicML](https://github.com/mosaicml/diffusion)
* [Together](https://www.together.xyz/)

## Reports / news
* [Dont worry about the Vase](https://thezvi.wordpress.com/) also author of [lesswrong](https://www.lesswrong.com/)
* [THE AI INDEX REPORT](https://aiindex.stanford.edu/report/)
* [On the Opportunities and Risks of Foundation Models](https://crfm.stanford.edu/report.html)

## Foundational :) Papers
* [Chain of Hindsight Aligns Language Models with Feedback](https://arxiv.org/abs/2302.02676) 2023
* [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) (2023)
* [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf) 2023
* [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf) (2023)
* [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/abs/2303.16199) (2023)
* [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf) (2023)
* [Scaling Transformer to 1M tokens and beyond with RMT](https://arxiv.org/abs/2304.11062)[code](https://github.com/booydar/t5-experiments/tree/scaling-report) (2023)
* [Will we run out of data? An analysis of the limits of scaling datasets in Machine Learning](https://arxiv.org/pdf/2211.04325.pdf)(2022)
* [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)(2022)
* [SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions](https://arxiv.org/pdf/2212.10560.pdf)
* [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)(2021)
* [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf)(2023)
* [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/pdf/2303.12712.pdf)
* [Choose Your Weapon: Survival Strategies for Depressed AI Academics](https://arxiv.org/pdf/2304.06035.pdf)
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (2023)
* [What Language Model to Train if You Have One Million GPU Hours?](https://arxiv.org/abs/2210.15424)(2022)
* [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) (2021)
* [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2021)
* [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) (2019)
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) (2020)
* [Language Models are Few-Shot Learners GPT3](https://arxiv.org/pdf/2005.14165.pdf) (2020)
* [Language Models are Unsupervised Multitask Learners GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2018)

### Surveys
* [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/pdf/2304.13712v2.pdf) (2023)

## NLP Courses
* [Stanford's CS 324 - Advances in Foundation Models](https://stanford-cs324.github.io/winter2023/)
* [Mohit Iyyer @UMASS](https://people.cs.umass.edu/~miyyer/cs685/schedule.html)
* [Ashish Vaswani's lecture on AIAYN @ Stanford CS224N: NLP with Deep Learning](https://www.youtube.com/watch?v=5vcj8kSwBCY) (2019)

## Prompt
* [promptbase](https://promptbase.com/)
* [shareGPT](https://sharegpt.com/)

## People

From researchers that showed us how to embed words ([Tomáš Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en)) to attention models, to transformers, BERT, Text to Text, and finally to GPT and Foundational models.

* [Alec Radford](https://scholar.google.com/citations?user=dOad5HoAAAAJ&hl=en)
* [Ashish Vaswani](https://scholar.google.com/citations?user=oR9sCGYAAAAJ&hl=en)
* [Colin Raffel](https://scholar.google.com/citations?user=I66ZBYwAAAAJ&hl=en)
* [Jacob Devlin](https://www.semanticscholar.org/author/Jacob-Devlin/39172707)
* [Hyung Won Chung](https://scholar.google.com/citations?user=1CAlXvYAAAAJ&hl=en)
* [Ian Hogarth](https://www.ianhogarth.com/about)
* [Ilya Sutskever](https://scholar.google.com/citations?user=x04W_mMAAAAJ&hl=en)
* [Percy Liang](https://cs.stanford.edu/~pliang/)
* [Tomáš Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en)

## Investors
* [AGI blog @ a16z](https://a16z.com/tag/generative-ai/)
* [Ian Hogarth](https://pluralplatform.com/the-peers/ian-hogarth/)
* [Nathan Benaich](https://www.nathanbenaich.com/) @Air Street Capital
* [Sequoia](https://www.sequoiacap.com/article/generative-ai-a-creative-new-world/?itm_medium=related-content&itm_source=sequoiacap.com)

## Engineering Notes
* [Cookbook for solving common problems in building GPT/LLM apps](https://bootcamp.uxdesign.cc/cookbook-for-solving-common-problems-in-building-gpt-llm-apps-93fcdbe3f44a)

