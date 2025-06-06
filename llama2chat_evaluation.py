import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
import numpy as np
from trl import SFTTrainer, SFTConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from random import randrange
from functools import partial


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Llama-2 model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, 
                       default="llama2_chat_classification_trainsampled_3_epochs/",
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"], 
                       help="Device to run evaluation on")
    
    # Data parameters
    parser.add_argument("--test_path", type=str, default="Datasets/test.csv",
                       help="Path to test data CSV file")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Number of sequences to return")
    parser.add_argument("--do_sample", action="store_true", default=False,
                       help="Whether to use sampling for generation")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for processing test data")
    
    # Embedding model parameters
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model for embedding similarity")
    
    # Sampling parameters (for testing subset)
    parser.add_argument("--use_sampling", action="store_true", default=False,
                       help="Whether to sample a subset of test data")
    parser.add_argument("--sample_size", type=int, default=100,
                       help="Number of samples per label when using sampling")
    parser.add_argument("--sample_seed", type=int, default=42,
                       help="Random seed for sampling")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="evaluation_results/",
                       help="Directory to save evaluation results")
    parser.add_argument("--save_predictions", action="store_true", default=False,
                       help="Whether to save predictions to file")
    parser.add_argument("--save_responses", action="store_true", default=False,
                       help="Whether to save generated responses to file")
    
    # Debug parameters
    parser.add_argument("--show_examples", type=int, default=2,
                       help="Number of examples to show during evaluation")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose output")
    
    return parser.parse_args()


def construct_test_input(sentence1, sentence2):
    prompt = "<s>[INST] <<SYS>>You are a helpful, respectful and honest assistant. Always follow the instructions provided and answer honestly.<</SYS>>\n"
    prompt += "Consider the following two sentences:\n"
    prompt += "Sentence1: " + sentence1 +"\n"
    prompt += "Sentence2: " + sentence2 +"\n"
    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent.\n [/INST] "
    return prompt


def load_and_prepare_test_data(test_path, use_sampling=False, sample_size=100, sample_seed=42):
    print(f"Loading test data from: {test_path}")
    
    # Load test data
    test = pd.read_csv(test_path, encoding='utf-8')
    
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in test.columns:
        test = test.drop(columns=['Unnamed: 0'])
    
    # Sample data if requested
    if use_sampling:
        print(f"Sampling {sample_size} examples per label...")
        labels = ['contrasting', 'entailment', 'neutral', 'reasoning']
        sampled_test_data = pd.concat([
            test[test['label'] == label].sample(
                min(sample_size, len(test[test['label'] == label])), 
                random_state=sample_seed
            ) for label in labels
        ])
        
        # Reset the index to avoid KeyError issues
        sampled_test_data.reset_index(drop=True, inplace=True)
        test = sampled_test_data
        
        # Print the count of each label
        label_counts = test['label'].value_counts()
        print("Label distribution in sampled data:")
        print(label_counts)
    
    # Apply construct_test_input function
    test['text'] = test.apply(lambda x: construct_test_input(x['sentence1'], x['sentence2']), axis=1)
    
    print(f"Total test samples: {len(test)}")
    return test


def load_model_and_tokenizer(model_path, device):
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    model.to(device)
    
    # Create pipeline
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    
    print("Model and tokenizer loaded successfully")
    return model, tokenizer, generator, device


def generate_response(prompt, generator, max_new_tokens, num_return_sequences, do_sample):
    outputs = generator(
        prompt, 
        max_new_tokens=max_new_tokens, 
        num_return_sequences=num_return_sequences, 
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    response = outputs[0]['generated_text']
    
    # Clean up response
    stop_tag = '<</SY>'
    if stop_tag in response:
        response = response.split(stop_tag)[0]
    
    return response.strip()


def generate_responses(test_data, generator, batch_size, max_new_tokens, num_return_sequences, do_sample, verbose=False):
    print("Generating responses for test data...")
    
    responses = []
    total_batches = (len(test_data['text']) + batch_size - 1) // batch_size
    
    for i in range(0, len(test_data['text']), batch_size):
        batch_prompts = test_data['text'][i:i+batch_size]
        batch_responses = [
            generate_response(prompt, generator, max_new_tokens, num_return_sequences, do_sample) 
            for prompt in batch_prompts
        ]
        responses.extend(batch_responses)
        
        current_batch = i // batch_size + 1
        print(f"Processed batch {current_batch}/{total_batches}")
        
        if verbose and current_batch == 1:
            print(f"Sample response from batch 1: {batch_responses[0][:200]}...")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    print(f"Generated {len(responses)} responses")
    return responses


def show_examples(test_data, generated_responses, num_examples):
    print(f"\n=== Showing {num_examples} Example(s) ===")
    
    for i in range(min(num_examples, len(test_data))):
        prompt = test_data['text'].iloc[i]
        response = generated_responses[i]
        true_label = test_data['label'].iloc[i]
        
        print(f"\nExample {i+1}:")
        print(f"True Label: {true_label}")
        print(f"Prompt:\n{prompt}")
        print(f"Generated Response:\n{response}")
        print("-" * 80)


def extract_relevant_response(response):
    last_inst_index = response.rfind("[/INST]")
    if last_inst_index != -1:
        relevant_part = response[last_inst_index + len("[/INST]"):].strip()
    else:
        relevant_part = response.strip()
    
    return relevant_part


def predict_labels_with_embeddings(generated_responses, embedding_model_name):
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Define label definitions
    label_to_definition = {
        "entailment": "Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.",
        "reasoning": "Sentence1 presents the reason, cause, or condition for the result or conclusion made Sentence2.",
        "contrasting": "Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.",
        "neutral": "Sentence1 and Sentence2 are independent."
    }
    
    definition_texts = list(label_to_definition.values())
    definition_embeddings = embedding_model.encode(definition_texts)
    
    print("Extracting relevant responses and predicting labels...")
    
    # Extract relevant responses
    relevant_responses = [extract_relevant_response(response) for response in generated_responses]
    
    # Predict labels using embeddings
    predicted_labels = []
    for relevant_response in relevant_responses:
        response_embedding = embedding_model.encode([relevant_response])
        similarities = cosine_similarity(response_embedding, definition_embeddings)
        best_match_index = similarities.argmax()
        predicted_labels.append(list(label_to_definition.keys())[best_match_index])
    
    print(f"Predicted {len(predicted_labels)} labels")
    return predicted_labels, relevant_responses


def calculate_metrics(true_labels, predicted_labels):
    print("Calculating evaluation metrics...")
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted')
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    
    return accuracy, precision_weighted, recall_weighted, f1_macro


def print_results(accuracy, precision_weighted, recall_weighted, f1_macro, true_labels, predicted_labels):
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision (Weighted): {precision_weighted:.4f}")
    print(f"Overall Recall (Weighted): {recall_weighted:.4f}")
    print(f"Overall F1 Score (Macro): {f1_macro:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        true_labels, 
        predicted_labels, 
        target_names=['Contrasting', 'Reasoning', 'Entailment', 'Neutral']
    ))


def save_results(test_data, predicted_labels, generated_responses, relevant_responses, 
                accuracy, precision_weighted, recall_weighted, f1_macro, 
                output_dir, save_predictions, save_responses):
    if not (save_predictions or save_responses):
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if save_predictions:
        # Save predictions
        results_df = pd.DataFrame({
            'sentence1': test_data['sentence1'],
            'sentence2': test_data['sentence2'],
            'true_label': test_data['label'],
            'predicted_label': predicted_labels,
            'correct': test_data['label'] == predicted_labels
        })
        
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        results_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
    
    if save_responses:
        # Save generated responses
        responses_df = pd.DataFrame({
            'sentence1': test_data['sentence1'],
            'sentence2': test_data['sentence2'],
            'true_label': test_data['label'],
            'predicted_label': predicted_labels,
            'full_response': generated_responses,
            'relevant_response': relevant_responses
        })
        
        responses_path = os.path.join(output_dir, 'generated_responses.csv')
        responses_df.to_csv(responses_path, index=False)
        print(f"Generated responses saved to: {responses_path}")
    
    # Save metrics summary
    metrics_summary = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro
    }
    
    metrics_path = os.path.join(output_dir, 'metrics_summary.txt')
    with open(metrics_path, 'w') as f:
        f.write("Evaluation Metrics Summary\n")
        f.write("="*30 + "\n")
        for metric, value in metrics_summary.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Metrics summary saved to: {metrics_path}")


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set seed for reproducibility
    if args.use_sampling:
        set_seed(args.sample_seed)
    
    print("=== Llama-2 Model Evaluation ===")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding model: {args.embedding_model}")
    print("=" * 60)
    
    # Load and prepare test data
    test_data = load_and_prepare_test_data(
        args.test_path, 
        args.use_sampling, 
        args.sample_size, 
        args.sample_seed
    )
    
    # Show example of test input
    if args.show_examples > 0:
        print(f"\nExample test input:")
        print(test_data['text'].iloc[0])
        print()
    
    # Load model and tokenizer
    model, tokenizer, generator, device = load_model_and_tokenizer(args.model_path, args.device)
    
    # Generate responses
    generated_responses = generate_responses(
        test_data,
        generator,
        args.batch_size,
        args.max_new_tokens,
        args.num_return_sequences,
        False,
        args.verbose
    )
    
    # Show examples if requested
    if args.show_examples > 0:
        show_examples(test_data, generated_responses, args.show_examples)
    
    # Predict labels using embeddings
    predicted_labels, relevant_responses = predict_labels_with_embeddings(
        generated_responses, 
        args.embedding_model
    )
    
    # Calculate metrics
    accuracy, precision_weighted, recall_weighted, f1_macro = calculate_metrics(
        test_data['label'], 
        predicted_labels
    )
    
    # Print results
    print_results(
        accuracy, 
        precision_weighted, 
        recall_weighted, 
        f1_macro, 
        test_data['label'], 
        predicted_labels
    )
    
    # Save results
    save_results(
        test_data,
        predicted_labels,
        generated_responses,
        relevant_responses,
        accuracy,
        precision_weighted,
        recall_weighted,
        f1_macro,
        args.output_dir,
        args.save_predictions,
        args.save_responses
    )
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()