# [A MISMATCHED Benchmark for Scientific Natural Language Inference](https://www.arxiv.org/abs/2506.04603)
This repository contains the dataset and code for the ACL 2025 Findings paper "A MISMATCHED Benchmark for Scientific Natural Language Inference" **The dataset can be downloaded from [here](https://drive.google.com/file/d/15GSYcakchdRmKlNoWIigGbCoI7eQLAUR/view?usp=sharing).**

**If you face any difficulties while downloading the dataset, raise an issue in this repository or contact us at fshaik8@uic.edu.**

## Abstract
Scientific Natural Language Inference (NLI) is the task of predicting the semantic relation between a pair of sentences extracted from re- search articles. Existing datasets for this task are derived from various computer science (CS) domains, whereas non-CS domains are com- pletely ignored. In this paper, we introduce a novel evaluation benchmark for scientific NLI, called MISMATCHED. The new MIS- MATCHED benchmark covers three non-CS domains–PSYCHOLOGY, ENGINEERING, and PUBLIC HEALTH, and contains 2,700 human annotated sentence pairs. We establish strong baselines on MISMATCHED using both Pre- trained Small Language Models (SLMs) and Large Language Models (LLMs). Our best per- forming baseline shows a Macro F1 of only 78.17% illustrating the substantial headroom for future improvements. In addition to in- troducing the MISMATCHED benchmark, we show that incorporating sentence pairs having an implicit scientific NLI relation between them in model training improves their performance on scientific NLI.

## Dataset Description
We introduce [MISMATCHED](https://drive.google.com/file/d/15GSYcakchdRmKlNoWIigGbCoI7eQLAUR/view?usp=sharing), a novel evaluation benchmark for scientific Natural Language Inference (NLI) designed to test out-of-domain generalization. It is derived from research articles across three non-Computer Science (CS) domains: PUBLIC HEALTH, PSYCHOLOGY, and ENGINEERING. MISMATCHED consists exclusively of development and test sets, containing 300 and 2,400 human-annotated sentence pairs respectively. Importantly, MISMATCHED does not include its own training set and serves as an out-of-domain evaluation test-bed.


The construction of the MISMATCHED development and test sets involved a two-phase process. In the first phase, sentence pair candidates for the ENTAILMENT, CONTRASTING, and REASONING classes were automatically extracted from the source articles. This extraction utilized a distant supervision method that relies on explicit linking phrases (e.g., "However," "Therefore") which are indicative of the semantic relation between adjacent sentences. These linking phrases were subsequently removed from the second sentence after the initial automatic labeling. For the NEUTRAL class, non-adjacent sentences from the same paper were paired using specific strategies.

In the second phase, all candidate pairs underwent a rigorous manual annotation process conducted by domain experts hired via a crowd-sourcing platform. This step was crucial to ensure high-quality data and create a realistic evaluation benchmark. Only those sentence pairs for which the human-assigned gold label matched the label automatically assigned during the distant supervision phase were included in the final MISMATCHED development and test sets.

We refer the reader to Section 3 of our [paper](https://www.arxiv.org/abs/2506.04603), for an in-depth description of the dataset construction process, data sources, and detailed statistics.

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

### Baseline Results

We establish strong baselines on MISMATCHED using both Pre-trained Small Language Models (SLMs) and Large Language Models (LLMs). The table below shows Macro F1 scores (%) with standard deviations across different domains and overall performance.

![Alt text](Images/Baseline_Results.png?raw=False "Title")

## Model Training & Testing
### Requirements
```
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
torch==2.5.1
transformers==4.51.1
```

**Supported Models**

**Note**: To reproduce the zero-shot and few-shot evaluation results reported in our paper for the MisMatched test dataset, please use one of the following 4 models exactly as specified:.

- microsoft/Phi-3-medium-128k-instruct (default)
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Meta-Llama-3.1-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3

#### Zero-Shot Evaluation Script (MisMatched_Zero_Shot.py)
##### Basic Usage
```
python MisMatched_Zero_Shot.py --model_name <model_name> --test_data_path <path_to_test.tsv>
```
##### Complete Command with All Parameters
```
python MisMatched_Zero_Shot.py --model_name "microsoft/Phi-3-medium-128k-instruct" --test_data_path "MisMatched/test.tsv" --env_file "var.env" --batch_size 128 --max_new_tokens 40 --embedding_model "all-MiniLM-L6-v2" --embedding_batch_size 256 --similarity_threshold 0.25 --sample_size 100 --random_seed 42 --output_file "zero_shot_results.csv" --num_sample_prompts 2 --do_sample --temperature 0.7 --top_p 0.9 --show_sample_responses --save_predictions --log_level "INFO" --force_cpu
```

#### Few-Shot Evaluation Script (MisMatched_Few_Shot.py)
##### Basic Usage
```
python MisMatched_Few_Shot.py --model_name <model_name> --train_data_path <path_to_train.csv> --test_data_path <path_to_test.tsv>
```
##### Complete Command with All Parameters
```
python MisMatched_Few_Shot.py --model_name "microsoft/Phi-3-medium-128k-instruct" --train_data_path "sampled_SciNLI_pair1.csv" --test_data_path "MisMatched/test.tsv" --env_file "var.env" --load_in_8bit --batch_size 128 --max_new_tokens 40 --use_sampled_examples --few_shot_samples_per_label 4 --few_shot_seed 0 --embedding_model "all-MiniLM-L6-v2" --embedding_batch_size 256 --similarity_threshold 0.25 --num_sample_prompts 2 --do_sample --temperature 0.7 --top_p 0.9 --show_sample_responses --save_predictions --output_file "few_shot_results.csv" --log_level "INFO"
```

### Fine-tuning and Evaluation of Pre-trained Small Language Models

```
python nli_train_evaluate.py --data_dir <location of a directory containing the train.tsv, test.tsv and dev.tsv files> --output_dir <directory to save model and results> --model_type <'BERT', 'Sci_BERT', 'Sci_BERT_uncased', 'RoBERTa', 'RoBERTa_large', 'xlnet'> --batch_size <batch size> --num_epochs <number of epochs to train the model for> --epoch_patience <patience for early stopping> --device <device to run your experiment on> --seed <some random seed>
```

### Llama-2 Fine-tuning and Evaluation

#### Fine-tuning Script(llama2chat_finetune.py)

##### Basic Usage
```
python llama2chat_finetune.py --train_path <path_to_train.csv> --dev_path <path_to_dev.csv> --output_dir <output_directory> --save_model_path <final_model_path>
```
##### Complete Command with All Parameters
```
python llama2chat_finetune.py --train_path "Datasets/train.csv" --dev_path "Datasets/dev.csv" --model_name "meta-llama/Llama-2-13b-chat-hf" --max_memory_per_gpu "80GB" --load_in_4bit --bnb_4bit_use_double_quant --bnb_4bit_quant_type "nf4" --bnb_4bit_compute_dtype "bfloat16" --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_bias "none" --output_dir "output/" --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --learning_rate 2e-3 --num_train_epochs 3 --save_strategy "epoch" --eval_strategy "epoch" --load_best_model_at_end --fp16 --optim "adamw_bnb_8bit" --max_seq_length 1024 --save_model_path "llama2_chat_classification_trainsampled_3_epochs/" --config_file "config.json" --seed 42
```

#### Evaluation Script (llama2chat_evaluation.py)

##### Basic Usage
```
python llama2chat_evaluation.py --model_path <path_to_finetuned_model> --test_path <path_to_test.csv>
```
##### Complete Command with All Parameters
```
python llama2chat_evaluation.py --model_path "llama2_chat_classification_trainsampled_3_epochs/" --device "auto" --test_path "Datasets/test.csv" --max_new_tokens 50 --num_return_sequences 1 --batch_size 128 --embedding_model "all-MiniLM-L6-v2" --use_sampling --sample_size 100 --sample_seed 42 --output_dir "evaluation_results/" --save_predictions --save_responses --show_examples 2 --verbose
```

## License
MisMatched is licensed with Attribution-ShareAlike 4.0 International [(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

## Contact
Please contact us at fshaik8@uic.edu with any questions.
