# Large Language Models

Resources I have found useful in my journey of [AGI](https://knowyourmeme.com/memes/shoggoth-with-smiley-face-artificial-intelligence). Some other overviews:

* [The Practical Guides for Large Language Models](https://github.com/Mooler0410/LLMsPracticalGuide)


![Alt Text](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/survey-gif-test.gif)




## Data

* [chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications#fn1trot6ka6e2)

## LLM models

See [list](https://en.wikipedia.org/wiki/Large_language_model) on Wikipedia
* [Bloom](https://huggingface.co/bigscience/bloom)
  - [Bloom-LORA](https://github.com/linhduongtuan/BLOOM-LORA)
  - [Petals](https://github.com/bigscience-workshop/petals)
* [Llama](https://github.com/facebookresearch/llama) Fill this [Form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) to get access to the weights
* [lit-llama](https://github.com/Lightning-AI/lit-llama) Independent implementation of LLaMA that is fully open source under the Apache 2.0 license.
* [redpajama](https://www.together.xyz/blog/redpajama) 
* [stable LLM](https://github.com/stability-AI/stableLM/) (cc license) (see [blog](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models))

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

# Compositional
* [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
* [Chameleon](https://github.com/lupantech/chameleon-llm)
* [HuggingGPT](https://github.com/microsoft/JARVIS)

## Tools
* [dalai](https://github.com/cocktailpeanut/dalai)
* [FUTUREPEDIA](https://www.futurepedia.io/)AI Tools dir updated daily
* [langchain](https://python.langchain.com/en/latest/)
* [llamaindex](https://gpt-index.readthedocs.io/en/latest/index.html)
* [Semantic Kernel](https://github.com/microsoft/semantic-kernel) [hello sk](https://devblogs.microsoft.com/semantic-kernel/hello-world/)

## Vector DBs

* [Chroma](https://www.trychroma.com/)
* [FAISS](https://github.com/facebookresearch/faiss) (were doing this way back)
* [Pinecone](https://www.pinecone.io/)
* [Vectera](https://vectara.com/)
* [Weaviate](https://weaviate.io/)

## Backends
* [deepspeed](https://github.com/microsoft/DeepSpeed)
* [ray](https://www.ray.io/)
* [modal](https://modal.com/)
* pytorch-lightening
  - [comparisons](https://sebastianraschka.com/blog/2023/pytorch-faster.html) (pytorch, mixed precision, static graphs, deepspeed, fabric)
  - [Train 1T+ Model parameters](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html)

## General Resources
* [reentry](https://rentry.org/localmodelslinks)
* [Distributed AI Research Institute](https://www.dair-institute.org/)

## Companies
* [startups @builtwithgenai](https://airtable.com/shr6nfE9FOHp17IjG/tblL3ekHZfkm3p6YT)
* [cohere](https://cohere.com/)

## Democratizaters
* [baseten](https://www.baseten.co/about)
* [huggingface](https://huggingface.co/)
* [Stable Diffusion Training with MosaicML](https://github.com/mosaicml/diffusion)

## Reports / news
* [Dont worry about the Vase](https://thezvi.wordpress.com/) also author of [lesswrong](https://www.lesswrong.com/)
* [THE AI INDEX REPORT](https://aiindex.stanford.edu/report/)
* [On the Opportunities and Risks of Foundation Models](https://crfm.stanford.edu/report.html)

## (Key) Papers

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
* [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) (2019)
* [Language Models are Few-Shot Learners GPT3](https://arxiv.org/pdf/2005.14165.pdf) (2020)
* [Language Models are Unsupervised Multitask Learners GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2018)

## NLP Courses
* [CS 324 - Advances in Foundation Models](https://stanford-cs324.github.io/winter2023/)
* [Mohit Iyyer @UMASS](https://people.cs.umass.edu/~miyyer/cs685/schedule.html)

## Prompt
* [promptbase](https://promptbase.com/)

## People

From researchers that showed us how to embed words ([Tomáš Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en)) to attention models, to transformers, BERT and finally to GPT and Foundational models.

* [Alec Radford](https://scholar.google.com/citations?user=dOad5HoAAAAJ&hl=en)
* [Ashish Vaswani](https://scholar.google.com/citations?user=oR9sCGYAAAAJ&hl=en)
* [Jacob Devlin](https://www.semanticscholar.org/author/Jacob-Devlin/39172707)
* [Hyung Won Chung](https://scholar.google.com/citations?user=1CAlXvYAAAAJ&hl=en)
* [Ian Hogarth](https://www.ianhogarth.com/about)
* [Ilya Sutskever](https://scholar.google.com/citations?user=x04W_mMAAAAJ&hl=en)
* [Percy Liang](https://cs.stanford.edu/~pliang/)
* [Tomáš Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en)

## VCs
* [AGI blog @ a16z](https://a16z.com/tag/generative-ai/)
* [Sequoia](https://www.sequoiacap.com/article/generative-ai-a-creative-new-world/?itm_medium=related-content&itm_source=sequoiacap.com)

