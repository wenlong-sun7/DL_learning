"""
Configuration file for the Mini GPT Language Model
Contains all hyperparameters and settings
"""

import os
import json
from typing import Dict, Any, Optional

# Model Configuration
VOCAB_SIZE = None  # Will be set dynamically based on vocabulary
EMBED_SIZE = 128  # Reduced from 256 for better training
NUM_LAYERS = 2    # Reduced from 4 for better training
NUM_HEADS = 4     # Increased from 2 for better attention
DROPOUT = 0.1

# Training Configuration
LEARNING_RATE = 0.01  # Increased from 0.001 for faster learning
NUM_EPOCHS = 20       # Increased from 10 for better convergence
MAX_OUTPUT_TOKENS = 10

# Debug Configuration
DEBUG = True  # Enable debug prints
SAVE_MODEL = True  # Save model after training

# Dataset Configuration
DATASET_FILE = None  # Must be specified by user
DATASETS_DIR = "datasets"        # Directory containing dataset files

# Special tokens
END_TOKEN = "<end>"
PAD_TOKEN = ""  # Empty string for padding


def load_training_data(dataset_file: str) -> Dict[str, str]:
    """
    Load training data from a dataset file
    
    Args:
        dataset_file: Name of the dataset file (e.g., 'simple_qa.json')
        
    Returns:
        Dictionary of training data
    """
    if not dataset_file:
        raise ValueError("Dataset file must be specified")
    
    # Construct full path to dataset file
    dataset_path = os.path.join(DATASETS_DIR, dataset_file)
    
    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"✅ Loaded dataset: {dataset_file}")
        print(f"📊 Dataset size: {len(training_data)} Q&A pairs")
        return training_data
        
    except Exception as e:
        raise RuntimeError(f"Error loading dataset {dataset_file}: {e}")


def list_available_datasets() -> list:
    """
    List all available dataset files
    
    Returns:
        List of available dataset filenames
    """
    if not os.path.exists(DATASETS_DIR):
        return []
    
    datasets = []
    for filename in os.listdir(DATASETS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(DATASETS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                datasets.append((filename, len(data)))
            except:
                continue
    
    return datasets


def set_dataset(dataset_file: str):
    """
    Set the dataset file to use for training
    
    Args:
        dataset_file: Name of the dataset file
    """
    global DATASET_FILE
    DATASET_FILE = dataset_file
    print(f"✅ Dataset set to: {dataset_file}")


def get_training_data(dataset_file: str) -> Dict[str, str]:
    """
    Get training data from a specific dataset file
    
    Args:
        dataset_file: Dataset file to load from
        
    Returns:
        Dictionary of training data
    """
    return load_training_data(dataset_file) 