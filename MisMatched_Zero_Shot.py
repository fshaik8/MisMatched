import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
import argparse
import sys


def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Model MisMatched Zero Shot Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    supported_models = [
        "microsoft/Phi-3-medium-128k-instruct",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-medium-128k-instruct",
        choices=supported_models,
        help="Name of the model to use from supported models"
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
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for text generation"
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
    
    # Sampling arguments
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples per label to use (for testing). If None, uses all data"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
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

def construct_test_input_phi3(sentence1, sentence2):
    prompt = "<|user|>\n"
    prompt += "You are a helpful assistant.\n"
    prompt += "Consider the following two sentences:\n"
    prompt += f"Sentence1: {sentence1}\n"
    prompt += f"Sentence2: {sentence2}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent."
    prompt += "<|end|>\n"
    prompt += "<|assistant|>"
    return prompt


def construct_test_input_llama2(sentence1, sentence2):
    prompt = "<s>[INST] <<SYS>>\n"
    prompt += "<</SYS>>\n"
    prompt += "Consider the following two sentences:\n"
    prompt += f"Sentence1: {sentence1}\n"
    prompt += f"Sentence2: {sentence2}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent.\n"
    prompt += "[/INST] "
    return prompt


def construct_test_input_llama3(sentence1, sentence2):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    prompt += "You are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
    prompt += "Consider the following two sentences:\n"
    prompt += f"Sentence1: {sentence1}\n"
    prompt += f"Sentence2: {sentence2}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent."
    prompt += "<|eot_id|>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>"
    return prompt


def construct_test_input_mistral(sentence1, sentence2):
    prompt = "<s>[INST]\n"
    prompt += "You are a helpful assistant.\n"
    prompt += "Consider the following two sentences:\n"
    prompt += f"Sentence1: {sentence1}\n"
    prompt += f"Sentence2: {sentence2}\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies, or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made in Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent."
    prompt += "[/INST]"
    return prompt


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


def extract_relevant_response_llama31(response):
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


MODEL_REGISTRY = {
    "microsoft/Phi-3-medium-128k-instruct": {
        "construct_test_input": construct_test_input_phi3,
        "extract_relevant_response": extract_relevant_response_phi3
    },
    "meta-llama/Llama-2-13b-chat-hf": {
        "construct_test_input": construct_test_input_llama2,
        "extract_relevant_response": extract_relevant_response_llama2
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "construct_test_input": construct_test_input_llama3,
        "extract_relevant_response": extract_relevant_response_llama31
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "construct_test_input": construct_test_input_mistral,
        "extract_relevant_response": extract_relevant_response_mistral
    }
}


def get_model_functions(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name]


def load_model(model_name, force_cpu=False):
    if force_cpu:
        max_memory = None
        device_map = "cpu"
        torch_dtype = None
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            max_memory = {i: str(torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)) + "GB" for i in range(n_gpus)}
            device_map = "auto"
            torch_dtype = "auto"
        else:
            max_memory = None
            device_map = "cpu"
            torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def check_model_precision(model, logger):
    dtypes = set()
    for param in model.parameters():
        dtypes.add(param.dtype)
    for buffer in model.buffers():
        dtypes.add(buffer.dtype)
    logger.info(f"Model weights and buffers are in the following data types: {dtypes}")


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
    
    if args.model_name == "meta-llama/Llama-2-13b-chat-hf" and args.batch_size == 128:
        args.batch_size = 64
        print(f"Adjusted batch size to {args.batch_size} for {args.model_name}")
    
    logger = setup_logging(args.log_level)
    
    model_functions = get_model_functions(args.model_name)
    construct_test_input = model_functions["construct_test_input"]
    extract_relevant_response = model_functions["extract_relevant_response"]
    
    logger.info(f"Using model: {args.model_name}")
    
    load_dotenv(args.env_file)
    
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token is None:
        raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable.")
    login(token)
    
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, args.force_cpu)
    
    check_model_precision(model, logger)
    
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    test_path = os.environ.get("MISMATCHED_TEST_DATA_PATH", args.test_data_path)
    logger.info(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path, sep='\t', encoding="utf-8")
    
    if args.sample_size is not None:
        logger.info(f"Sampling {args.sample_size} samples per label")
        test = test.groupby('label', group_keys=False).sample(
            n=args.sample_size, 
            random_state=args.random_seed
        ).reset_index(drop=True)
    
    test["text"] = test.apply(
        lambda x: construct_test_input(x["sentence1"], x["sentence2"]), axis=1
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
    
    # Setup embedding model
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
    
    # Overall metrics
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
        
        output_path = args.output_file if args.output_file else "predictions_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()