# Relevance Baseline - Cosine Similarity Threshold Optimization

### Description
This folder contains a Jupyter Notebook implementing a baseline approach for relevance classification using cosine similarity thresholds. The method involves classifying instances based on fixed thresholds for cosine similarity between vector representations of prompts and texts. The thresholds are optimized using the train set to maximize Kendall's Tau, a performance metric.

### Algorithm Details
In the process of optimizing threshold values, we iterate through three loops, each representing a threshold. This allows us to create a grid with various threshold combinations, ranging from the minimum to the maximum similarity value with a step of 0.025.

For each iteration, instances in the train set are classified based on the current thresholds:
- `least_relevant`: Similarity smaller than the first threshold.
- `second_least_relevant`: Similarity between the first and second thresholds.
- `second_most_relevant`: Similarity between the second and third thresholds.
- `most_relevant`: Similarity greater than the third threshold.

