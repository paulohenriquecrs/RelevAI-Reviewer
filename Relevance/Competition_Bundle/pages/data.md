# Data Collection Process
***
The data collection process consistst of the following steps:

* Collecting data from Semantic Scholar
* Parsing the papers (PDFs) with Grobid
* Reverse Engineering of Content to Extract Prompts
* Structuring the data in required format

# Data Description
***

The dataset contains five features, including prompt, most relevant, second most relevant, second least relevant, and least relevant, and they are described as below.
* **prompt**: This is the reverse-engineered prompt derived from the original paper. It encapsulates the core research question or topic that the paper addresses.
* **most_relevant**: This category includes the Title, Abstract, and Related Work sections of the original paper. 
* **second_most_relevant**: This involves the Title, Abstract, and Related Work from a paper cited by the original paper. This paper is likely to be relevant to the prompt.
* **second_least_relevant**: This category comprises the Title, Abstract, and Related Work from a paper randomly sampled from the same field as the original paper.
*  **least_relevant**: This includes the Title, Abstract, and Related Work from a paper randomly selected from a different field than the original. Its relevance to the prompt is expected to be the lowest.

### Sample data
**prompt**: Write a systematic survey or overview about the impact of individual behavior on social structure, including the influence of cultural norms, economic systems, and political institutions. Consider the ways in which individuals both shape and are shaped by the larger social framework in which they exist.

**most_relevant**: 
{'*title*': 'Individuals and Social Structure',
'*abstract*': 'Treatments of contextual effects in the social science literature have traditionally focused on statistical phenomena more than on social processes...',
'*related work'*: 'The existing literature on contextual effects in social science research has primarily focused on statistical phenomena rather than social processes. This approach often relies on identifying "group-level" effects as evidence of contextual processes, neglecting the underlying mechanisms through which social structure and social interaction impact individuals...'}

**second_most_relevant**:
{'*title*': 'Assessing School Effects: Some Identities.', 
'*abstract*': 'Two methods of assessing school effects-the contextual variables method and the analysis of covariance-are discussed within the framework of a general school effects model...'}

**second_least_relevant**:
{'*title*': 'Modeling Organizational Adaptation as a Simulated Annealing Process',
'*abstract*': "Organizations can be characterized as complex systems composed of adaptive and intelligent agents. Organizational adaptation occurs through restructuring and learning...", 
'*related work*': 'The related work section provides an overview of existing literature that is relevant to the research paper titled "Modeling Organizational Adaptation as a Simulated Annealing Process."'}

**least_relevant**:
{'*title*': 'Andreas Vesalius as a renaissance innovative neuroanatomist: his 5th centenary of birth.', 
'*abstract*': 'Andreas Vesalius (1514-1564) is considered the Father of Modern Anatomy, and an authentic representative of the Renaissance...', 
'*related work*': 'Andreas Vesalius, a significant figure in the field of medical history, has been extensively studied and analyzed in numerous publications...'}