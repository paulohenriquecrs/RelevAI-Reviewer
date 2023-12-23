# Relevance Data

***

## Data

We have provided you a sample data `relevance_sample_data.csv`, to experiemnt with and use it the way you want. Read the following sections to understand the data collection process and dataset description.

## Data Collection Process

The data collection process consistst of the following steps:

- Collecting data from Semantic Scholar 
- Parsing the papers (PDFs) with Grobid 
- Reverse Engineering of Content to Extract Prompts
- Structuring the data in required format

## Data Description

Each row is a paper where we only keep Title+Abstract+Related work (produced from chatGPT based on their citations). We categorize the relevance of papers based on their relationship to an original paper's prompt. The dataset features are given below:

- `prompt`: This is the reverse-engineered prompt derived from the original paper. It encapsulates the core research question or topic that the paper addresses.

- `most_relevant`: This category includes the Title, Abstract, and Related Work sections of the original paper.

- `second_most_relevant`: This involves the Title, Abstract, and Related Work from a paper cited by the original paper. This paper is likely to be relevant to the prompt.

- `second_least_relevant`: This category comprises the Title, Abstract, and Related Work from a paper randomly sampled from the same field as the original paper.

- `least_relevant`: This includes the Title, Abstract, and Related Work from a paper randomly selected from a different field than the original paper. Its relevance to the prompt is expected to be the lowest.

## Getting Started with Semantic Scholar and OpenAI API (Optional)

- The Semantic Scholar API facilitates access to a vast database of scholarly articles. It offers endpoints for retrieving details about papers, authors, and more. Find detailed documentation here: [Semantic Scholar API Documentation](https://api.semanticscholar.org/api-docs/).
- For enhanced access and more frequent API calls, obtain an API key by filling out this form: [Request API Key](https://www.semanticscholar.org/product/api).
