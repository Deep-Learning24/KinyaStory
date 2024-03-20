# Project Proposal: Storytelling Model in Kinyarwanda Language

### Getting started

To get started with the project, you need to install the required packages. You can do this by executing the following script:

```bash
python setup.py
```

## 1. Introduction

Preserving traditional cultural narratives is essential amid modernization. While technology can pose a threat to some cultural aspects, it also offers opportunities for creative arts, precisely generative art. This work introduces a Deep Learning model tailored for storytelling in Kinyarwanda language. We propose building a text-generating model aiming at preserving Rwandan heritage. Using deep learning methodologies, our aim is to capture and replicate the essence of the old written narratives, ensuring that this cherished art form endures for generations to come.

## 2. Literature Review

The work of researchers from Eindhoven University of Technology and the University of Amsterdam in the Netherlands, as well as Greenhouse Group B.V., on a generative Approach to Story Narration[2]. This model can generate stories and complementary illustrated images with short prompt text inputs from users.

KinyaBERT[1], a BERT variant good at capturing morphological compositionality and expressing word-relative syntactic regularities for Kinyarwanda. This model is designed for rich morphological languages, which is both low-resource and has complex word forms. KinyaBERT is better at understanding the structure of words and sentences in Kinyarwanda, even when there are translation errors. Overall, it’s a step forward in improving language models for less common languages which are rich in morphology.

## 3. Source of Data

The project will utilize an open-sourced dataset of Kinyarwanda text available on public domain. We collect data using a web application [Data collection] for easy collecting and handling our dataset. The dataset structure can be viewed from here. This consists of various genres: fictions (novel, and novella), historical stories, and drama/plays to ensure the model's versatility and robustness. Data preprocessing techniques will be applied to clean and standardize the dataset for training purposes.

## 4. Baseline Model

The unsupervised sub-word tokenization methods commonly used in BERT variant models (e.g., byte-pair encoding – BPE) are sub-optimal at handling morphologically rich languages, especially low resource languages like Kinyarwanda. We will use KinyaBERT architecture[1], a simple two-tier BERT for morphologically rich languages pretrained on Kinyarwanda text. We will also consider two other baseline models Storytelling AI[2] which use small GPT-2 fine-tuned on English dataset of 100 short stories written by the Brothers Grimm as well as A text generation and prediction system pre trained using BERT and GPT-2[3] that uses OpenAI GPT-2 to generate sentences based on the starting words and BERT to generate predictions for the 𝑚𝑎𝑠𝑘ed label position in a given sentence. The model can make predictions based on the words around the label.

## 5. Proposed Solution

The proposed solution involves enhancing the base model's performance on Kinyarwanda text through several strategies:

- Fine-tuning the model on Kinyarwanda data set of short, medium and long stories.
- Incorporating additional layers or modules to improve the model's ability to generate coherent and contextually relevant text.
- Evaluating the model's performance using metrics BLEU score, and human evaluation to assess its fluency, coherence, and semantic accuracy.

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