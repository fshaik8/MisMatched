import os
import argparse
import logging
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from sklearn.utils import shuffle

import transformers
from transformers import (
    BertTokenizer, 
    RobertaTokenizer, 
    XLNetTokenizer,
    AutoTokenizer
)

import logging as python_logging
from Training_and_testing_utils_transformers import train_model, test_model
from bert_data_preprocessor import create_data_for_bert
from Utils import load_data


def setup_logging(log_level: str = "INFO") -> None:
    python_logging.basicConfig(
        level=getattr(python_logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_label_mappings() -> Dict[str, int]:
    return {
        'contrasting': 0,
        'reasoning': 1,
        'entailing': 2,
        'neutral': 3,
    }


def get_numeric_label(label: str, label_to_idx_dict: Dict[str, int]) -> int:
    return label_to_idx_dict.get(label, -1)


def convert_first_letter_to_lower(text: str) -> str:
    if not text or len(text) == 0:
        return text
    return text[0].lower() + text[1:]


def get_tokenizer(model_type: str, cache_dir: Optional[str] = None):
    tokenizer_configs = {
        'Sci_BERT': ('allenai/scibert_scivocab_cased', BertTokenizer, False),
        'Sci_BERT_uncased': ('allenai/scibert_scivocab_uncased', BertTokenizer, False),
        'RoBERTa': ('roberta-base', RobertaTokenizer, False),
        'RoBERTa_large': ('roberta-large', RobertaTokenizer, False),
        'xlnet': ('xlnet-base-cased', XLNetTokenizer, False),
        'BERT': ('bert-base-cased', BertTokenizer, False),
    }
    
    if model_type not in tokenizer_configs:
        python_logging.warning(f"Unknown model type {model_type}, defaulting to BERT")
        model_type = 'BERT'
    
    model_name, tokenizer_class, do_lower_case = tokenizer_configs[model_type]
    
    try:
        tokenizer = tokenizer_class.from_pretrained(
            model_name, 
            do_lower_case=do_lower_case,
            cache_dir=cache_dir
        )
        python_logging.info(f"Loaded tokenizer for {model_type}")
        return tokenizer
    except Exception as e:
        python_logging.error(f"Failed to load tokenizer for {model_type}: {e}")
        raise


def load_datasets(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    
    files = {
        'train': 'train.tsv',
        'dev': 'dev.tsv',
        'test': 'test.tsv'
    }
    
    try:
        train_df = pd.read_csv(data_path / files['train'], sep='\t')
        dev_df = pd.read_csv(data_path / files['dev'], sep='\t')
        test_df = pd.read_csv(files['test'], sep='\t')
        
        python_logging.info(f"Loaded datasets - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
        return train_df, dev_df, test_df
        
    except FileNotFoundError as e:
        python_logging.error(f"Dataset file not found: {e}")
        raise
    except Exception as e:
        python_logging.error(f"Error loading datasets: {e}")
        raise


def preprocess_datasets(
    train_df: pd.DataFrame, 
    dev_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    label_to_idx_dict: Dict[str, int],
    apply_lowercase: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Standardize column names
    column_mappings = {
        'gold_label': 'label',
        'first_sentence': 'sentence1',
        'second_sentence': 'sentence2'
    }
    
    for df in [train_df, dev_df, test_df]:
        df.rename(columns=column_mappings, inplace=True)
    
    # Check if any dataset has 'entailment' values in the label column and update them to 'entailing'
    for df_name, df in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        if 'label' in df.columns:
            entailment_count = (df['label'] == 'entailment').sum()
            if entailment_count > 0:
                python_logging.info(f"Found {entailment_count} 'entailment' labels in {df_name} dataset. Updating to 'entailing'.")
                df['label'] = df['label'].replace('entailment', 'entailing')
    
    # Apply lowercase conversion if specified
    if apply_lowercase:
        for df in [train_df, dev_df, test_df]:
            if 'sentence2' in df.columns:
                df['sentence2'] = df['sentence2'].apply(convert_first_letter_to_lower)
    
    # Convert labels to numeric
    for df in [train_df, dev_df, test_df]:
        df['label'] = df['label'].apply(lambda x: get_numeric_label(x, label_to_idx_dict))
        # Filter out invalid labels
        df = df[df['label'] >= 0]
    
    # Remove rows with invalid labels
    train_df = train_df[train_df['label'] >= 0].reset_index(drop=True)
    dev_df = dev_df[dev_df['label'] >= 0].reset_index(drop=True)
    test_df = test_df[test_df['label'] >= 0].reset_index(drop=True)
    
    python_logging.info(f"After preprocessing - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    return train_df, dev_df, test_df


def prepare_data(
    output_dir: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    tokenizer,
    model_type: str,
    label_to_idx_dict: Dict[str, int]
) -> None:
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    python_logging.info("Creating tokenized data...")
    
    output_location = str(output_path) + '/'
    
    # Updated function calls with error handling
    try:
        python_logging.info("Processing training data...")
        create_data_for_bert(output_location, label_to_idx_dict, train_df, tokenizer, 'train', model_type)
    except Exception as e:
        python_logging.error(f"Error processing training data: {e}")
        raise
    
    try:
        python_logging.info("Processing test data...")
        create_data_for_bert(output_location, label_to_idx_dict, test_df, tokenizer, 'test', model_type)
    except Exception as e:
        python_logging.error(f"Error processing test data: {e}")
        raise
    
    try:
        python_logging.info("Processing validation data...")
        create_data_for_bert(output_location, label_to_idx_dict, dev_df, tokenizer, 'valid', model_type)
    except Exception as e:
        python_logging.error(f"Error processing validation data: {e}")
        raise
    
    # Debug: Check what files were actually created
    python_logging.info("Files created in output directory:")
    for file_path in output_path.rglob("*.pkl"):
        python_logging.info(f"  Found: {file_path}")
    
    python_logging.info("Data preparation completed")


def load_prepared_data(data_dir: str) -> Tuple[np.ndarray, ...]:
    data_path = Path(data_dir)
    
    try:
        X_train = np.asarray(load_data(str(data_path / 'X_train.pkl')))
        X_test = np.asarray(load_data(str(data_path / 'X_test.pkl')))
        X_valid = np.asarray(load_data(str(data_path / 'X_valid.pkl')))
        
        att_mask_train = load_data(str(data_path / 'att_mask_train.pkl'))
        att_mask_test = load_data(str(data_path / 'att_mask_test.pkl'))
        att_mask_valid = load_data(str(data_path / 'att_mask_valid.pkl'))
        
        y_train = np.asarray(load_data(str(data_path / 'y_train.pkl')))
        y_test = np.asarray(load_data(str(data_path / 'y_test.pkl')))
        y_valid = np.asarray(load_data(str(data_path / 'y_valid.pkl')))
        
        python_logging.info(f"Loaded data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, X_valid, att_mask_train, att_mask_test, att_mask_valid, y_train, y_test, y_valid
        
    except Exception as e:
        python_logging.error(f"Error loading prepared data: {e}")
        raise


def train_and_evaluate(
    model_dir: str,
    model_type: str,
    X_train: np.ndarray,
    att_mask_train: List,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    att_mask_valid: List,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    att_mask_test: List,
    y_test: np.ndarray,
    config: Dict,
    test_df: Optional[pd.DataFrame] = None
) -> None:
    
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    python_logging.info("Starting model training...")

    train_model(
        str(model_path), model_type, X_train, att_mask_train, y_train,
        X_valid, att_mask_valid, y_valid, config['device'],
        config['batch_size'], config['accumulation_steps'], config['num_epochs'],
        config['num_classes'], config['report_every'], config['epoch_patience'],
        learning_rate=config['learning_rate']
    )
    
    python_logging.info("Training completed. Starting evaluation...")
    
    # Evaluate model
    test_result_file = model_path / config['test_result_file']
    
    test_model(
        str(model_path), model_type, X_test, att_mask_test, y_test,
        config['device'], config['batch_size'], config['num_classes'],
        True, str(test_result_file), test_df
    )
    
    python_logging.info(f"Evaluation completed. Results saved to {test_result_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train NLI models with transformers')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the SciNLI dataset files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save model and results')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, 
                       choices=['Sci_BERT', 'Sci_BERT_uncased', 'RoBERTa', 'RoBERTa_large', 'xlnet', 'BERT'],
                       default='Sci_BERT', help='Model type to use')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Directory to cache downloaded models')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epoch_patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--report_every', type=int, default=10, help='Report frequency during training')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA device IDs')
    parser.add_argument('--apply_lowercase', action='store_true',
                       help='Apply lowercase to second sentences')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Setup
    setup_logging(args.log_level)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    set_random_seeds(args.seed)
    
    start_time = datetime.datetime.now()
    python_logging.info(f"Starting training at {start_time}")
    
    try:
        # Get label mappings for SciNLI
        label_to_idx_dict = get_label_mappings()
        num_classes = len(label_to_idx_dict)
        
        # Load datasets
        train_df, dev_df, test_df = load_datasets(args.data_dir)
        
        # Check if test dataset has Domain column
        has_domain_column = 'Domain' in test_df.columns
        if has_domain_column:
            python_logging.info("Test dataset contains Domain column - domain-specific evaluation will be performed")
        else:
            python_logging.info("Test dataset does not contain Domain column - only overall evaluation will be performed")
        
        # Preprocess datasets
        train_df, dev_df, test_df = preprocess_datasets(
            train_df, dev_df, test_df, label_to_idx_dict, args.apply_lowercase
        )
        
        # Get tokenizer
        tokenizer = get_tokenizer(args.model_type, args.cache_dir)
        
        # Prepare output directory
        prepared_data_dir = Path(args.output_dir) / args.model_type / 'prepared_data'
        model_dir = Path(args.output_dir) / args.model_type / 'model'
        
        # Prepare data
        prepare_data(
            str(prepared_data_dir), train_df, dev_df, test_df,
            tokenizer, args.model_type, label_to_idx_dict
        )
        
        # Load prepared data
        X_train, X_test, X_valid, att_mask_train, att_mask_test, att_mask_valid, y_train, y_test, y_valid = load_prepared_data(str(prepared_data_dir))
        
        # Training configuration
        config = {
            'batch_size': args.batch_size,
            'accumulation_steps': args.accumulation_steps,
            'num_epochs': args.num_epochs,
            'num_classes': num_classes,
            'report_every': args.report_every,
            'epoch_patience': args.epoch_patience,
            'device': args.device,
            'learning_rate': args.learning_rate,
            'test_result_file': 'test_performance.csv'
        }
        
        # Train and evaluate - pass test_df if it has Domain column
        train_and_evaluate(
            str(model_dir), args.model_type,
            X_train, att_mask_train, y_train,
            X_valid, att_mask_valid, y_valid,
            X_test, att_mask_test, y_test,
            config, test_df if has_domain_column else None
        )
        
        end_time = datetime.datetime.now()
        python_logging.info(f"Training completed at {end_time}")
        python_logging.info(f"Total time: {end_time - start_time}")
        
    except Exception as e:
        python_logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()