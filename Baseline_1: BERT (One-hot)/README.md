# Relevance Baseline 1: BERT with One-hot encoding

### Description
This baseline implementation utilizes BERT (Bidirectional Encoder Representations from Transformers) for sequence classification with one hot encoded labels to predict the relevance class of each text. 

### Methodology
In our analysis using classification models, our primary focus is on comparing traditional machine learning models against a more expensive model, namely BERT.

#### BERT (Bidirectional Encoder Representations from Transformers)
BERT is a pre-trained transformer-based model introduced by Devlin et al. at Google in 2018 [1]. It excels in natural language processing tasks by capturing deep contextual relationships bidirectionally within text data, leading to state-of-the-art performance across various NLP applications. 

In our experiment, we employed the BERT sequence classification approach to learn the relevance ranks. Particularly, we leveraged the bert-based-uncase pre-trained model [2], which consists of 12 layers, a hidden size of 768, and 12 self-attention heads, totaling 110 million parameters, for tokenization and training.

### References
[1] Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

[2] Hugging Face Model Hub. (n.d.). Google's BERT Model. https://huggingface.co/google-bert/bert-base-uncased
