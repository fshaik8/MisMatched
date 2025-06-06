import os
import argparse
import json
from random import randrange
from functools import partial
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
from huggingface_hub import login


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-2")
    
    # Data parameters
    parser.add_argument("--train_path", type=str, default="Datasets/train.csv",
                       help="Path to training data CSV file")
    parser.add_argument("--dev_path", type=str, default="Datasets/dev.csv",
                       help="Path to validation data CSV file")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf",
                       help="Pre-trained model name from Hugging Face Hub")
    parser.add_argument("--max_memory_per_gpu", type=str, default="80GB",
                       help="Maximum memory allocation per GPU")
    
    # Quantization parameters
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Load model in 4-bit precision")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=False,
                       help="Use double quantization for 4-bit")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                       choices=["nf4", "fp4"], help="4-bit quantization type")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"], help="Compute dtype for 4-bit quantization")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout rate")
    parser.add_argument("--lora_bias", type=str, default="none",
                       choices=["none", "all", "lora_only"], help="LoRA bias configuration")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="output/",
                       help="Output directory for model checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                       help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of updates steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-3,
                       help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--save_strategy", type=str, default="epoch",
                       choices=["no", "epoch", "steps"], help="Save strategy")
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                       choices=["no", "epoch", "steps"], help="Evaluation strategy")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True,
                       help="Load best model at end of training")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--optim", type=str, default="adamw_bnb_8bit",
                       help="Optimizer to use")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                       help="Maximum sequence length for training")
    
    # Model saving parameters
    parser.add_argument("--save_model_path", type=str, 
                       default="llama2_chat_classification_trainsampled_3_epochs/",
                       help="Path to save the final fine-tuned model")
    
    # Authentication parameters
    parser.add_argument("--config_file", type=str, default="config.json",
                       help="Path to configuration file containing Hugging Face token")
    
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def huggingface_login(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            token = config.get('huggingface_token')
        
        if token:
            login(token)
            print("Successfully logged into Hugging Face Hub")
        else:
            print("Warning: No Hugging Face token found in config file")
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found. Proceeding without authentication.")
    except Exception as e:
        print(f"Warning: Error during Hugging Face login: {e}")


def construct_input(sentence1, sentence2, label):
    label_to_answer = {
        "contrasting": "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.",
        "reasoning": "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made Sentence2.",
        "entailment": "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.",
        "neutral": "d. Sentence1 and Sentence2 are independent."
    }

    prompt = "<s>[INST] <<SYS>>You are a helpful, respectful and honest assistant. Always follow the instructions provided and answer honestly.<</SYS>>\n"
    prompt += "Consider the following two sentences:\n"
    prompt += "Sentence1: " + sentence1 +"\n"
    prompt += "Sentence2: " + sentence2 +"\n"

    prompt += "Based on only the information available in these two sentences, which of the following options is true?\n"
    prompt += "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
    prompt += "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made Sentence2.\n"
    prompt += "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
    prompt += "d. Sentence1 and Sentence2 are independent.\n [/INST] "

    prompt += "Based on the information available in the two sentences, the correct answer is " + label_to_answer[label]

    return prompt


def load_and_prepare_data(train_path, dev_path):
    print(f"Loading training data from: {train_path}")
    print(f"Loading validation data from: {dev_path}")
    
    # Load the data
    train = pd.read_csv(train_path, encoding='utf-8')
    dev = pd.read_csv(dev_path, encoding='utf-8')

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in train.columns:
        train = train.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in dev.columns:
        dev = dev.drop(columns=['Unnamed: 0'])

    # Apply the construct_input function to the data
    train['text'] = train.apply(lambda x: construct_input(x['sentence1'], x['sentence2'], x['label']), axis=1)
    dev['text'] = dev.apply(lambda x: construct_input(x['sentence1'], x['sentence2'], x['label']), axis=1)

    # Convert the Pandas DataFrames to Hugging Face Datasets
    train_data = Dataset.from_pandas(train)
    dev_data = Dataset.from_pandas(dev)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(dev_data)}")
    
    return train_data, dev_data


def create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    compute_dtype = dtype_mapping.get(bnb_4bit_compute_dtype, torch.bfloat16)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    return bnb_config


def load_model(model_name, bnb_config, max_memory_per_gpu):
    print(f"Loading model: {model_name}")
    
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # Allocate max memory for each GPU
    max_memory = {i: max_memory_per_gpu for i in range(n_gpus)}

    # Load model and set up device mapping to GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Distribute across GPUs if available
        max_memory=max_memory
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

    # Set padding token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure the model has a pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Model and tokenizer loaded successfully")
    return model, tokenizer


def create_peft_config(lora_r, lora_alpha, lora_dropout, lora_bias):
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM",
    )
    return peft_config


def create_training_arguments(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        fp16=args.fp16,
        optim=args.optim
    )
    return training_args


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print("=== Llama-2 Fine-tuning ===")
    print(f"Model: {args.model_name}")
    print(f"Training epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)
    
    # Hugging Face Hub Login
    huggingface_login(args.config_file)
    
    # Load and prepare data
    train_data, dev_data = load_and_prepare_data(args.train_path, args.dev_path)
    
    # Create BitsAndBytes configuration
    bnb_config = create_bnb_config(
        args.load_in_4bit,
        args.bnb_4bit_use_double_quant,
        args.bnb_4bit_quant_type,
        args.bnb_4bit_compute_dtype
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name, bnb_config, args.max_memory_per_gpu)
    
    # Create PEFT configuration
    peft_config = create_peft_config(
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_bias
    )
    
    # Create training arguments
    training_args = create_training_arguments(args)
    
    # Enable input requires_grad and gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Ensure use_cache is set to False
    model.config.use_cache = False
    
    print("Initializing trainer...")
    # Initialize trainer
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_data,
        dataset_text_field="text",
        eval_dataset=dev_data,
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    print("Starting training...")
    # Start training
    trainer.train()
    
    print(f"Saving model to: {args.save_model_path}")
    # Save the model
    trainer.save_model(args.save_model_path)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()