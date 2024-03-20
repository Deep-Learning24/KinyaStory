# PROJECT: Storytelling Model for Kinyarwanda Language

## 1. 
Train and test the GPT-2 model on the ROCstories dataset. 
The dataset comprises three key components: ROCStories winter 2017,
ROCStories spring 2016, and the Story Cloze Test Winter 2018 (excluding validation)
These datasets offer a diverse range of complete five-sentence stories
and scenario-based tests, providing rich narrative contexts and challenging the model’s
comprehension abilities. By combining stories from different seasons and formats, 
the model gains exposure to varied storytelling styles, themes, and lengths, fostering adaptability
and preventing biases. This comprehensive dataset composition aims to train a robust text generation model capable of producing engaging and coherent narratives across a wide spectrum of scenarios.


## 4. Baseline Model

We use a Transformer (Vaswani et al., 2017) based architecture for our LMs. 
The model largely follows the details of the OpenAI GPT model (Radford et al., 2018)
with a few modifications. Layer normalization (Ba et al., 2016) was moved to the input
of each sub-block, similar to a pre-activation residual network (He et al., 2016) and an
additional layer normalization was added after the final selfattention block. A modified 
initialization which accountsfor the accumulation on the residual path with model depth
is used. We scale the weights of residual layers at initialization by a factor of 1/√N 
where N is the number of residual layers. The vocabulary is expanded to 50,257. 
Wealso increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.


## 6. Deliverables

The project will deliver the following outcomes:

- A text generating model tailored for Kinyarwanda language.
- Documentation detailing the model architecture, training process, and evaluation results.
- A comparative analysis between the baseline models and the improved Kinyarwanda-specific model, highlighting performance gains and areas for further improvement.
- Codebase and resources made available as open source to encourage collaboration and future research in Kinyarwanda NLP.

## References

[1] A. Nzeyimana and A. Niyongabo Rubungo, “KinyaBERT: a Morphology-aware Kinyarwanda Language Model,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds., Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 5347–5363. doi: 10.18653/v1/2022.acl-long.367.

[2] S. Fotedar, K. Vannisselroij, S. Khalil, and B. Ploeg, “Storytelling AI: A Generative Approach to Story Narration”.

[3] Y. Qu, P. Liu, W. Song, L. Liu, and M. Cheng, “A Text Generation and Prediction System: Pre-training on New Corpora Using BERT and GPT-2,” in 2020 IEEE 10th International Conference on Electronics Information and Emergency Communication (ICEIEC), Jul. 2020, pp. 323–326. doi: 10.1109/ICEIEC49280.2020.9152352.