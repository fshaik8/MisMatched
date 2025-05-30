# A MISMATCHED Benchmark for Scientific Natural Language Inference
This repository contains the dataset and code for the ACL 2025 Findings paper "A MISMATCHED Benchmark for Scientific Natural Language Inference" **The dataset can be downloaded from [here](https://drive.google.com/file/d/15GSYcakchdRmKlNoWIigGbCoI7eQLAUR/view?usp=sharing).**

**If you face any difficulties while downloading the dataset, raise an issue in this repository or contact us at fshaik8@uic.edu.**

## Abstract
Scientific Natural Language Inference (NLI) is the task of predicting the semantic relation between a pair of sentences extracted from re- search articles. Existing datasets for this task are derived from various computer science (CS) domains, whereas non-CS domains are com- pletely ignored. In this paper, we introduce a novel evaluation benchmark for scientific NLI, called MISMATCHED. The new MIS- MATCHED benchmark covers three non-CS domains–PSYCHOLOGY, ENGINEERING, and PUBLIC HEALTH, and contains 2,700 human annotated sentence pairs. We establish strong baselines on MISMATCHED using both Pre- trained Small Language Models (SLMs) and Large Language Models (LLMs). Our best per- forming baseline shows a Macro F1 of only 78.17% illustrating the substantial headroom for future improvements. In addition to in- troducing the MISMATCHED benchmark, we show that incorporating sentence pairs having an implicit scientific NLI relation between them in model training improves their performance on scientific NLI.

## Dataset Description
We introduce [MISMATCHED](https://drive.google.com/file/d/15GSYcakchdRmKlNoWIigGbCoI7eQLAUR/view?usp=sharing), a novel evaluation benchmark for scientific Natural Language Inference (NLI) designed to test out-of-domain generalization. It is derived from research articles across three non-Computer Science (CS) domains: PUBLIC HEALTH, PSYCHOLOGY, and ENGINEERING. MISMATCHED consists exclusively of development and test sets, containing 300 and 2,400 human-annotated sentence pairs respectively. Importantly, MISMATCHED does not include its own training set and serves as an out-of-domain evaluation test-bed.


The construction of the MISMATCHED development and test sets involved a two-phase process. In the first phase, sentence pair candidates for the ENTAILMENT, CONTRASTING, and REASONING classes were automatically extracted from the source articles. This extraction utilized a distant supervision method that relies on explicit linking phrases (e.g., "However," "Therefore") which are indicative of the semantic relation between adjacent sentences. These linking phrases were subsequently removed from the second sentence after the initial automatic labeling. For the NEUTRAL class, non-adjacent sentences from the same paper were paired using specific strategies.

In the second phase, all candidate pairs underwent a rigorous manual annotation process conducted by domain experts hired via a crowd-sourcing platform. This step was crucial to ensure high-quality data and create a realistic evaluation benchmark. Only those sentence pairs for which the human-assigned gold label matched the label automatically assigned during the distant supervision phase were included in the final MISMATCHED development and test sets.

We refer the reader to Section 3 of our paper(), for an in-depth description of the dataset construction process, data sources, and detailed statistics.

### Examples
![Alt text](Images/Examples.png?raw=False "Title")

### Files
The MISMATCHED dataset provides development and test data in Tab-Separated Value (.tsv) format:

  => test.tsv and dev.tsv contain the testing and development data, respectively. Each of these .tsv files includes the following columns essential for the Natural Language Inference task: 
  
    * sentence1: The premise sentence..
    
    * sentence2: The hypothesis sentence
    
    * label: The human-annotated label representing the semantic relation (e.g., ENTAILMENT, CONTRASTING, REASONING, NEUTRAL).

    Additionally, the files contain metadata columns: Domain which allows for domain-specific analyses.
  
### Dataset Size (MISMATCHED Benchmark)
The MISMATCHED benchmark is specifically designed as an out-of-domain evaluation set and does not include a training set. The sizes for the provided, human-annotated sets are as follows:

  * Test: 2,400 sentence pairs - human annotated.

  * Dev: 300 sentence pairs - human annotated.

  * Total (Dev + Test): 2,700 sentence pairs.

## License
MisMatched is licensed with Attribution-ShareAlike 4.0 International [(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

## Contact
Please contact us at fshaik8@uic.edu with any questions.
