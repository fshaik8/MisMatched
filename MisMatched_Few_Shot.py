import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support,
)
import logging
from huggingface_hub import login
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
import argparse
import sys


def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_model_family(model_name):
    if "Phi-3" in model_name:
        return "phi3"
    elif "Llama-2" in model_name:
        return "llama2"
    elif "Llama-3" in model_name or "Meta-Llama-3" in model_name:
        return "llama3"
    elif "Mistral" in model_name or "mistral" in model_name:
        return "mistral"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_default_batch_size(model_family: str) -> int:
    return 128 if model_family == "llama3" else 32


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Model Few Shot Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-medium-128k-instruct",
        choices=[
            "microsoft/Phi-3-medium-128k-instruct",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ],
        help="Name of the model to use from Hugging Face"
    )
    
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=True,
        help="Load model in 8-bit precision using BitsAndBytes"
    )
    
    parser.add_argument(
        "--no_8bit",
        action="store_true",
        help="Disable 8-bit loading (overrides --load_in_8bit)"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="sampled_SciNLI_pair1.csv",
        help="Path to the training/few-shot examples CSV file"
    )
    
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="MisMatched/test.tsv",
        help="Path to the test data TSV file"
    )
    
    parser.add_argument(
        "--env_file",
        type=str,
        default="var.env",
        help="Path to the environment file containing tokens"
    )
    
    # Few-shot configuration
    parser.add_argument(
        "--use_sampled_examples",
        action="store_true",
        help="Use sampled few-shot examples instead of full training data"
    )
    
    parser.add_argument(
        "--few_shot_samples_per_label",
        type=int,
        default=4,
        help="Number of few-shot examples per label (only used with --use_sampled_examples)"
    )
    
    parser.add_argument(
        "--few_shot_seed",
        type=int,
        default=0,
        help="Random seed for few-shot example sampling"
    )
    
    # Generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for text generation (auto-selected based on model if not specified)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="Maximum number of new tokens to generate"
    )
    
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling (only used if do_sample is True)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p value for nucleus sampling (only used if do_sample is True)"
    )
    
    # Embedding model arguments
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings"
    )
    
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=256,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.25,
        help="Threshold for similarity matching"
    )
    
    # Display arguments
    parser.add_argument(
        "--num_sample_prompts",
        type=int,
        default=2,
        help="Number of sample prompts to display"
    )
    
    parser.add_argument(
        "--show_sample_responses",
        action="store_true",
        help="Whether to show sample responses during execution"
    )
    
    # Device arguments
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the results CSV file"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save predictions to a file"
    )
    
    return parser.parse_args()


def create_bnb_config(load_in_8bit):
    bnb_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    return bnb_config


def load_model(model_name, bnb_config, force_cpu=False):
    if force_cpu:
        max_memory = None
        device_map = "cpu"
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            max_memory = {i: str(torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)) + "GB" for i in range(n_gpus)}
            device_map = "auto"
        else:
            max_memory = None
            device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        max_memory=max_memory,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def map_label_to_option(label):
    label_to_option = {
        "contrasting": ("c", "Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1."),
        "reasoning": ("b", "Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2."),
        "entailing": ("a", "Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2."),
        "neutral": ("d", "Sentence1 and Sentence2 are independent.")
    }
    return label_to_option[label]


def construct_few_shot_input_phi3(example_pairs, sentence1, sentence2):
    def construct_user_input(sentence1, sentence2):
        return (
            "<|user|>\n"
            f"Consider the following two sentences:\n"
            f"Sentence1: {sentence1}\n"
            f"Sentence2: {sentence2}\n"
            "Based on only the information available in these two sentences, which of the following options is true?\n"
            "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
            "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
            "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
            "d. Sentence1 and Sentence2 are independent.<|end|>\n"
        )

    def construct_assistant_response(option, explanation):
        return (
            "<|assistant|>\n"
            f"Based on the information available in the two sentences, the correct answer is ({option}) {explanation}<|end|>\n"
        )

    prompt = ""
    for _, row in example_pairs.iterrows():
        label = row['label']
        sentence1_example = row['sentence1']
        sentence2_example = row['sentence2']
        option, explanation = map_label_to_option(label)
        prompt += construct_user_input(sentence1_example, sentence2_example)
        prompt += construct_assistant_response(option, explanation)
    prompt += construct_user_input(sentence1, sentence2) + "<|assistant|>\n"
    return prompt


def construct_few_shot_input_llama2(example_pairs, sentence1, sentence2):
    prompt = "<s>[INST] <<SYS>>\n"
    prompt += "<</SYS>>\n"

    first_row = example_pairs.iloc[0]
    label = first_row['label']
    sentence1_example = first_row['sentence1']
    sentence2_example = first_row['sentence2']
    option, explanation = map_label_to_option(label)

    prompt += f"Consider the following two sentences:\nSentence1: {sentence1_example}\n"
    prompt += f"Sentence2: {sentence2_example}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent.\n"

    prompt += " [/INST] "
    prompt += f"Based on the information available in the two sentences, the correct answer is ({option}) {explanation}"
    prompt += " </s>\n"

    for i, row in enumerate(example_pairs.iloc[1:].iterrows(), start=2):
        label = row[1]['label']
        sentence1_example = row[1]['sentence1']
        sentence2_example = row[1]['sentence2']
        option, explanation = map_label_to_option(label)

        prompt += "<s>[INST]\n"
        prompt += f"Consider the following two sentences:\nSentence1: {sentence1_example}\n"
        prompt += f"Sentence2: {sentence2_example}\n"
        prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
        prompt += "a. Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2.\n"
        prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
        prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
        prompt += "d. Sentence1 and Sentence2 are independent.\n"
        prompt += " [/INST] "
        prompt += f"Based on the information available in the two sentences, the correct answer is ({option}) {explanation}"
        prompt += " </s>\n"

    prompt += "<s>[INST]\n"
    prompt += f"Consider the following two sentences:\nSentence1: {sentence1}\n"
    prompt += f"Sentence2: {sentence2}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent.\n"
    prompt += " [/INST]"
    
    return prompt


def construct_few_shot_input_llama3(example_pairs, sentence1, sentence2):
    def construct_user_input(sentence1, sentence2):
        return (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Consider the following two sentences:\nSentence1: {sentence1}\n"
            f"Sentence2: {sentence2}\n"
            "Based on only the information available in these two sentences, which of the following options is true?\n"
            "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
            "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
            "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
            "d. Sentence1 and Sentence2 are independent.<|eot_id|>\n"
        )
    
    def construct_assistant_response(option, explanation):
        return (
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"Based on the information available in the two sentences, the correct answer is ({option}) {explanation}"
            "<|eot_id|>\n"
        )
    
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>\n"
    )
    for _, row in example_pairs.iterrows():
        label = row['label']
        sentence1_example = row['sentence1']
        sentence2_example = row['sentence2']
        option, explanation = map_label_to_option(label)
        prompt += construct_user_input(sentence1_example, sentence2_example)
        prompt += construct_assistant_response(option, explanation)
    prompt += construct_user_input(sentence1, sentence2) + "<|start_header_id|>assistant<|end_header_id|>"
    return prompt


def construct_few_shot_input_mistral(example_pairs, sentence1, sentence2):
    prompt = "<s>[INST]\n"
    prompt += "You are a helpful assistant.\n"
    first_row = example_pairs.iloc[0]
    label = first_row['label']
    sentence1_example = first_row['sentence1']
    sentence2_example = first_row['sentence2']
    option, explanation = map_label_to_option(label)

    prompt += f"Consider the following two sentences:\nSentence1: {sentence1_example}\n"
    prompt += f"Sentence2: {sentence2_example}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent."
    prompt += " [/INST]"

    prompt += f" Based on the information available in the two sentences, the correct answer is ({option}) {explanation}"
    prompt += "</s>\n"

    for i, row in enumerate(example_pairs.iloc[1:].iterrows(), start=2):
        label = row[1]['label']
        sentence1_example = row[1]['sentence1']
        sentence2_example = row[1]['sentence2']
        option, explanation = map_label_to_option(label)

        prompt += "<s>[INST]\n"
        prompt += f"Consider the following two sentences:\nSentence1: {sentence1_example}\n"
        prompt += f"Sentence2: {sentence2_example}\n"
        prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
        prompt += "a. Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2.\n"
        prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
        prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
        prompt += "d. Sentence1 and Sentence2 are independent."
        prompt += " [/INST]"

        prompt += f" Based on the information available in the two sentences, the correct answer is ({option}) {explanation}"
        prompt += "</s>\n"

    prompt += "<s>[INST]\n"
    prompt += f"Consider the following two sentences:\nSentence1: {sentence1}\n"
    prompt += f"Sentence2: {sentence2}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent."
    prompt += " [/INST]"
    return prompt


def get_construct_few_shot_input_function(model_family):
    functions = {
        "phi3": construct_few_shot_input_phi3,
        "llama2": construct_few_shot_input_llama2,
        "llama3": construct_few_shot_input_llama3,
        "mistral": construct_few_shot_input_mistral
    }
    return functions[model_family]


def extract_relevant_response_phi3(response):
    last_end_header_index = response.rfind("<|assistant|>")
    if last_end_header_index != -1:
        relevant_part = response[last_end_header_index + len("<|assistant|>"):].strip()
    else:
        relevant_part = response.strip()
    return relevant_part


def extract_relevant_response_llama2(response):
    last_end_header_index = response.rfind("[/INST]")
    if last_end_header_index != -1:
        relevant_part = response[last_end_header_index + len("[/INST]"):].strip()
    else:
        relevant_part = response.strip()
    return relevant_part


def extract_relevant_response_llama3(response):
    last_end_header_index = response.rfind("<|end_header_id|>")
    if last_end_header_index != -1:
        relevant_part = response[last_end_header_index + len("<|end_header_id|>"):].strip()
    else:
        relevant_part = response.strip()
    return relevant_part


def extract_relevant_response_mistral(response):
    last_end_header_index = response.rfind("[/INST]")
    if last_end_header_index != -1:
        relevant_part = response[last_end_header_index + len("[/INST]"):].strip()
    else:
        relevant_part = response.strip()
    return relevant_part


def get_extract_relevant_response_function(model_family):
    functions = {
        "phi3": extract_relevant_response_phi3,
        "llama2": extract_relevant_response_llama2,
        "llama3": extract_relevant_response_llama3,
        "mistral": extract_relevant_response_mistral
    }
    return functions[model_family]


def evaluate_predictions(true_labels, predicted_labels, possible_labels, logger):
    valid_indices = [i for i, label in enumerate(predicted_labels) if label is not None]
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    filtered_predicted_labels = [predicted_labels[i] for i in valid_indices]

    accuracy = accuracy_score(filtered_true_labels, filtered_predicted_labels)
    precision_weighted = precision_score(
        filtered_true_labels,
        filtered_predicted_labels,
        average="weighted",
        labels=possible_labels,
        zero_division=0,
    )
    recall_weighted = recall_score(
        filtered_true_labels,
        filtered_predicted_labels,
        average="weighted",
        labels=possible_labels,
        zero_division=0,
    )
    f1_macro = f1_score(
        filtered_true_labels,
        filtered_predicted_labels,
        average="macro",
        labels=possible_labels,
        zero_division=0,
    )

    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Overall Precision (Weighted): {precision_weighted:.4f}")
    logger.info(f"Overall Recall (Weighted): {recall_weighted:.4f}")
    logger.info(f"Overall F1 Score (Macro): {f1_macro:.4f}")

    report = classification_report(
        filtered_true_labels,
        filtered_predicted_labels,
        target_names=possible_labels,
        zero_division=0,
    )
    logger.info("\nEvaluation metrics for each label class:\n" + report)


def generate_responses(test_data, generator, batch_size, max_new_tokens, do_sample, temperature, top_p, logger):
    responses = []
    total_batches = (len(test_data) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch_prompts = test_data["text"].iloc[i : i + batch_size].tolist()
        try:
            batch_outputs = generator(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                batch_size=len(batch_prompts),
            )
            batch_responses = [outputs[0]['generated_text'] for outputs in batch_outputs]
            responses.extend([response.strip() for response in batch_responses])
            logger.info(f"Processed batch {i // batch_size + 1}/{total_batches}")
        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
            responses.extend([""] * len(batch_prompts))
        finally:
            torch.cuda.empty_cache()
    return responses


def main():
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    load_dotenv(args.env_file)
    
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token is None:
        raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable.")
    login(token)

    model_family = get_model_family(args.model_name)
    logger.info(f"Detected model family: {model_family}")
    
    if args.batch_size is None:
        args.batch_size = get_default_batch_size(model_family)
        logger.info(f"Auto-selected batch size: {args.batch_size}")

    load_in_8bit = args.load_in_8bit and not args.no_8bit
    logger.info(f"Loading model in 8-bit: {load_in_8bit}")
    
    bnb_config = create_bnb_config(load_in_8bit)
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, bnb_config, args.force_cpu)
    
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    construct_few_shot_input = get_construct_few_shot_input_function(model_family)
    extract_relevant_response = get_extract_relevant_response_function(model_family)
    
    train_path = os.environ.get("SCINLI_FEW_SHOT_PAIRS_PATH", args.train_data_path)
    logger.info(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path, encoding='utf-8')
    
    if args.use_sampled_examples:
        logger.info(f"Using sampled few-shot examples with seed: {args.few_shot_seed}")
        random.seed(args.few_shot_seed)
        
        labels = ['entailment', 'reasoning', 'contrasting', 'neutral']
        few_shot_examples = pd.concat(
            [train[train['label'] == label].sample(args.few_shot_samples_per_label, random_state=args.few_shot_seed) 
             for label in labels if label in train['label'].values],
            ignore_index=True
        )
        
        few_shot_examples['label'].replace({'entailment': 'entailing'}, inplace=True)
        logger.info(f"Few-shot examples labels:\n{few_shot_examples[['label']]}")
        examples_for_prompt = few_shot_examples
    else:
        train['label'].replace({'entailment': 'entailing'}, inplace=True)
        logger.info(f"Training data labels:\n{train[['label']]}")
        examples_for_prompt = train
    
    test_path = os.environ.get("MISMATCHED_TEST_DATA_PATH", args.test_data_path)
    logger.info(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path, sep='\t', encoding="utf-8")
    
    logger.info("Constructing few-shot prompts...")
    test['text'] = test.apply(
        lambda x: construct_few_shot_input(examples_for_prompt, x['sentence1'], x['sentence2']), axis=1
    )
    
    print("Sample Prompt:")
    print(f"Prompt 1:\n{test['text'].iloc[0]}\n{'-'*80}\n")
    
    if args.show_sample_responses:
        print("Sample Prompts and Generated Responses:")
        for i in range(min(args.num_sample_prompts, len(test))):
            prompt = test['text'].iloc[i]
            print(f"Prompt {i+1}:\n{prompt}\n{'-'*80}")
            response = generator(
                prompt,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=1,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p
            )[0]['generated_text']
            print(f"Generated Response {i+1}:\n{response}\n{'='*80}\n")
    
    logger.info("Generating responses...")
    generated_responses = generate_responses(
        test, generator, args.batch_size, args.max_new_tokens,
        args.do_sample, args.temperature, args.top_p, logger
    )
    
    relevant_responses = [extract_relevant_response(response) for response in generated_responses]
    del generated_responses

    embedding_device = 'cpu' if args.force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading embedding model: {args.embedding_model} on {embedding_device}")
    embedding_model = SentenceTransformer(args.embedding_model, device=embedding_device)

    label_to_definition = {
        "entailing": "Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.",
        "reasoning": "Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.",
        "contrasting": "Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.",
        "neutral": "Sentence1 and Sentence2 are independent."
    }
    
    definition_texts = list(label_to_definition.values())
    definition_embeddings = embedding_model.encode(
        definition_texts,
        batch_size=args.embedding_batch_size,
        convert_to_tensor=True
    )
    
    logger.info("Encoding responses...")
    response_embeddings = embedding_model.encode(
        relevant_responses,
        batch_size=args.embedding_batch_size,
        convert_to_tensor=True
    )
    del relevant_responses
    
    logger.info("Predicting labels using similarity matching...")
    predicted_labels = []
    similarities = cosine_similarity(response_embeddings.cpu(), definition_embeddings.cpu())
    best_match_scores = similarities.max(axis=1)
    best_match_indices = similarities.argmax(axis=1)
    
    for score, index in zip(best_match_scores, best_match_indices):
        if score >= args.similarity_threshold:
            predicted_labels.append(list(label_to_definition.keys())[index])
        else:
            predicted_labels.append(None)
    
    possible_labels = ["entailing", "reasoning", "contrasting", "neutral"]
    evaluate_predictions(test["label"].tolist(), predicted_labels, possible_labels, logger)
    
    del response_embeddings, similarities
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    domain_metrics = defaultdict(dict)
    
    if 'Domain' not in test.columns:
        raise ValueError("The 'Domain' column is not present in the test data.")
    
    domains = set(test['Domain'])
    
    for domain in domains:
        domain_true_labels = test.loc[test['Domain'] == domain, 'label']
        domain_indices = domain_true_labels.index
        domain_generated_labels = [predicted_labels[i] for i in domain_indices]
        
        valid_indices = [i for i, label in enumerate(domain_generated_labels) if label is not None]
        filtered_true_labels = [domain_true_labels.iloc[i] for i in valid_indices]
        filtered_generated_labels = [domain_generated_labels[i] for i in valid_indices]
        
        if len(filtered_true_labels) == 0:
            continue
        
        accuracy = accuracy_score(filtered_true_labels, filtered_generated_labels)
        
        precision_macro, recall_macro, f1_macro_function, _ = precision_recall_fscore_support(
            filtered_true_labels,
            filtered_generated_labels,
            average='macro',
            zero_division=0
        )
        
        domain_metrics[domain] = {
            "Accuracy": accuracy,
            "F1_Macro": f1_macro_function,
            "Precision (Macro)": precision_macro,
            "Recall (Macro)": recall_macro
        }
    
    overall_true_labels = [test["label"].iloc[i] for i, label in enumerate(predicted_labels) if label is not None]
    overall_predicted_labels = [label for label in predicted_labels if label is not None]
    
    overall_accuracy = accuracy_score(overall_true_labels, overall_predicted_labels)
    overall_precision_macro, overall_recall_macro, overall_f1_macro_function, _ = precision_recall_fscore_support(
        overall_true_labels,
        overall_predicted_labels,
        average='macro',
        zero_division=0
    )
    
    domain_metrics["Overall"] = {
        "Accuracy": overall_accuracy,
        "F1_Macro": overall_f1_macro_function,
        "Precision (Macro)": overall_precision_macro,
        "Recall (Macro)": overall_recall_macro,
    }
    
    metrics_df = pd.DataFrame(domain_metrics).T
    print("\nDomain-Specific Classification Report:")
    print(metrics_df)
    
    if args.save_predictions or args.output_file:
        results_df = test.copy()
        results_df['predicted_label'] = predicted_labels
        
        output_path = args.output_file if args.output_file else "few_shot_predictions_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
    
    logger.info("Few-shot evaluation completed successfully!")


if __name__ == "__main__":
    main()