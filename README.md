# RelevAI-Reviewer

This repository contains the code and resources for the RelevAI-Reviewer system, a tool designed to automate the review of scientific papers by assessing their relevance to a given prompt. Originally developed as part of the Creation of an AI Challenge class project in the Artificial Intelligence Master at Université Paris-Saclay, the system has since evolved into a research paper submitted to a conference.

### Abstract
Recent advancements in Artificial Intelligence (AI), particularly the widespread adoption of Large Language Models (LLMs), have significantly enhanced text analysis capabilities. This technological evolution offers considerable promise for automating the review of scientific papers, a task traditionally managed through peer review by fellow researchers. Despite its critical role in maintaining research quality, the conventional peer-review process is often slow and subject to biases, potentially impeding the swift propagation of scientific knowledge. In this paper, we propose RelevAI-Reviewer, an automatic system that conceptualizes the task of survey paper review as a classification challenge, aimed at assessing the relevance of a paper in relation to a specified prompt, analogous to a "call for papers". To address this, we introduce a novel dataset comprised of prompts and four papers, each varying in relevance to the prompt. The objective is to develop a machine learning (ML) model capable of determining the relevance of each paper and identifying the most pertinent one. We explore various baseline approaches, including traditional ML classifiers like Support Vector Classifier (SVC) and advanced language models such as BERT. Preliminary findings indicate that the BERT-based end-to-end classifier surpasses other conventional ML methods in performance. We present this problem as a public challenge to foster engagement and interest in this area of research.

### Dataset
The full dataset used in this research can be accessed [here](https://drive.google.com/drive/u/1/folders/1fG74aCrU43J7gJvTyQaai2AlpN8dYHKL?usp=sharing_eip_m&invite=CLuInRE&ts=65a4f341).

### Authors
#### Students
- Paulo Henrique Couto (Group Leader)
- Nageeta Kumari 
- Quang Phuoc HO

#### Supervisors
- Lisheng Sun-Hosoya
- Benedictus Kent Rachmat
- Ihsan Ullah

### Competition on Codabench
We believe the AI-assisted reviewer holds considerable potential for practical implementation. The work presented in this paper serves as an initial investigation. To engage the AI community and foster further research in this domain, we are introducing this problem as an open challenge, where we provide access to the Relevance-AI datasets and our baseline models' code. We invite interested parties to participate. For more details and to join the challenge, please visit the challenge website [here](https://www.codabench.org/competitions/1946/).

For more details, please refer to the paper associated with this project:

- Paulo Henrique Couto: paulo.couto-de-resende-silva@universite-paris-saclay.fr
