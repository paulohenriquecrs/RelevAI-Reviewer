# Soundness Data

***

## Data

We have provided you the following sample data files to experiemnt with and use it the way you want. Read the following sections to understand the data collection process and dataset description.

- `soundness_sample_data_train.csv`
- `soundness_sample_data_test.csv`

## Data Collection Process

The data collection process consistst of the following steps:

- Collecting data from Semantic Scholar
- Parsing the papers (PDFs) with Grobid
- Filtering Papers with specific format of citations
- Extracting context sentences and their citation/reference
- Extracting reference paper abstract from Semantic Scholar
- Structuring the data in required format

## Data Description

Each row is a paper where we only keep Title+Abstract+Related work (produced from chatGPT based on their citations). We categorize the relevance of papers based on their relationship to an original paper's prompt. The dataset features are given below:

- `context_sentences`: This is centence from a paper which cites a paper

- `citation_number`: This is the citation number i.e. 1,2,3, etc. which is written in the context

- `paper_title`: Title of the paper from which context_sentence is taken

- `paper_id`: Semantic Scholar paper id for the paper from which context_sentence is taken

- `paper_PDF_url`: URL of the paper from which context_sentence is taken

- `reference_title`: Title of the reference paper

- `reference_paper_id`: Paper id of the reference paper

- `reference_abstract`: Abstract of the reference paper

- `label`: *1* if the reference is cited in the context, *0* otherwise

## Getting Started with Semantic Scholar and OpenAI API (Optional)

- The Semantic Scholar API facilitates access to a vast database of scholarly articles. It offers endpoints for retrieving details about papers, authors, and more. Find detailed documentation here: [Semantic Scholar API Documentation](https://api.semanticscholar.org/api-docs/).
- For enhanced access and more frequent API calls, obtain an API key by filling out this form: [Request API Key](https://www.semanticscholar.org/product/api).
