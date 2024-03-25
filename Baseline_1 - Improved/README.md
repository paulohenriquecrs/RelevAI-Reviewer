# Relevance Baseline 1 (Improved with Thermometer Encoding)

### Description
This improved version of Baseline 1 utilizes BERT (Bidirectional Encoder Representations from Transformers) for sequence classification with thermometer encoded labels to predict the relevance class of each text.

### Methodology
In our analysis using classification models, our primary focus is on comparing traditional machine learning models against a more expensive model, namely BERT.

#### Thermometer Encoding
Thermometer encoding is a label encoding technique used to represent categorical targets, especially when dealing with ordinal data. In this approach, each bit of the label vector differentiates between specific classes. The encoding is designed to capture ordinal relationships between classes, allowing the model to learn the relative importance of different ranks.

We utilized thermometer encoding to represent the relevance ranks of papers to a given prompt. Each paper's relevance rank is represented as a thermometer vector, where the sum of the vector components indicates the overall relevance rank. Relevant ranks are assigned based on the descending order of the sums, with higher sums indicating higher relevance ranks.

we chose to use thermometer encoding to ensure that each of the four paper candidates evaluated against a prompt receives a unique relevance rank. By employing this encoding method, we guarantee that no two papers are classified in the same category of relevance. This approach helps us maintain a clear and distinct ranking for each paper, enhancing the precision and reliability of our classification process.

### References
[1] Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

[2] Hugging Face Model Hub. (n.d.). Google's BERT Model. https://huggingface.co/google-bert/bert-base-uncased
