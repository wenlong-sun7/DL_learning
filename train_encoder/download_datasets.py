"""
Dataset downloader for Mini GPT Language Model
Downloads and formats small datasets for training
"""

import os
import json
from typing import List, Dict, Any

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️  datasets library not installed. Install with: pip install datasets")
    DATASETS_AVAILABLE = False


def create_simple_qa_dataset():
    """Create a simple Q&A dataset manually"""
    return {
        "what is your name": "my name is mini gpt <end>",
        "how old are you": "i am a new model <end>",
        "what do you do": "i help answer questions <end>",
        "where do you live": "i live in the computer <end>",
        "what is the weather": "i cannot check the weather <end>",
        "what time is it": "i cannot tell time <end>",
        "what is your favorite color": "i like blue <end>",
        "what is your favorite food": "i do not eat food <end>",
        "can you help me": "yes i can try to help <end>",
        "what is machine learning": "it is teaching computers to learn <end>",
        "what is artificial intelligence": "it is making computers smart <end>",
        "what is python": "it is a programming language <end>",
        "what is javascript": "it is a web programming language <end>",
        "what is html": "it is a markup language for websites <end>",
        "what is css": "it is for styling websites <end>",
        "what is a database": "it stores information <end>",
        "what is an algorithm": "it is a step by step solution <end>",
        "what is a function": "it is a reusable piece of code <end>",
        "what is a variable": "it stores data in programming <end>",
        "what is a loop": "it repeats code multiple times <end>"
    }


def download_squad_sample(num_samples: int = 200) -> Dict[str, str]:
    """Download a sample from SQuAD dataset"""
    if not DATASETS_AVAILABLE:
        return create_simple_qa_dataset()
    
    try:
        dataset = load_dataset("squad", split=f"train[:{num_samples}]")
        qa_data = {}
        
        for item in dataset:
            question = item['question'].lower().strip()
            answer = item['answers']['text'][0].lower().strip()
            qa_data[question] = f"{answer} <end>"
        
        return qa_data
    except Exception as e:
        print(f"Error downloading SQuAD: {e}")
        return create_simple_qa_dataset()


def download_daily_dialog_sample(num_samples: int = 150) -> Dict[str, str]:
    """Download a sample from Daily Dialog dataset"""
    if not DATASETS_AVAILABLE:
        return create_simple_qa_dataset()
    
    try:
        dataset = load_dataset("daily_dialog", split=f"train[:{num_samples}]")
        qa_data = {}
        
        for item in dataset:
            dialog = item['dialog']
            if len(dialog) >= 2:
                question = dialog[0].lower().strip()
                answer = dialog[1].lower().strip()
                qa_data[question] = f"{answer} <end>"
        
        return qa_data
    except Exception as e:
        print(f"Error downloading Daily Dialog: {e}")
        return create_simple_qa_dataset()


def download_wikitext_sample(num_samples: int = 100) -> Dict[str, str]:
    """Download a sample from WikiText dataset"""
    if not DATASETS_AVAILABLE:
        return create_simple_qa_dataset()
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{num_samples}]")
        qa_data = {}
        
        for i, item in enumerate(dataset):
            text = item['text'].strip()
            if len(text) > 20 and len(text) < 200:  # Filter reasonable length
                # Create simple Q&A from text
                words = text.split()
                if len(words) >= 4:
                    question = f"what is {words[0]}"
                    answer = " ".join(words[:5])  # First 5 words as answer
                    qa_data[question] = f"{answer} <end>"
        
        return qa_data
    except Exception as e:
        print(f"Error downloading WikiText: {e}")
        return create_simple_qa_dataset()


def save_dataset(dataset: Dict[str, str], filename: str):
    """Save dataset to JSON file"""
    os.makedirs("datasets", exist_ok=True)
    filepath = os.path.join("datasets", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved to: {filepath}")
    print(f"📊 Dataset size: {len(dataset)} Q&A pairs")


def load_dataset_from_file(filename: str) -> Dict[str, str]:
    """Load dataset from JSON file"""
    filepath = os.path.join("datasets", filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Dataset file not found: {filepath}")
        return create_simple_qa_dataset()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Dataset loaded from: {filepath}")
    print(f"📊 Dataset size: {len(dataset)} Q&A pairs")
    return dataset


def main():
    """Main function to download and save datasets"""
    print("🚀 Mini GPT Dataset Downloader")
    print("="*50)
    
    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)
    
    # Download different datasets
    print("\n📥 Downloading datasets...")
    
    # 1. Simple Q&A dataset
    print("\n1️⃣ Creating simple Q&A dataset...")
    simple_qa = create_simple_qa_dataset()
    save_dataset(simple_qa, "simple_qa.json")
    
    # 2. SQuAD sample
    print("\n2️⃣ Downloading SQuAD sample...")
    squad_data = download_squad_sample(200)
    save_dataset(squad_data, "squad_sample.json")
    
    # 3. Daily Dialog sample
    print("\n3️⃣ Downloading Daily Dialog sample...")
    dialog_data = download_daily_dialog_sample(150)
    save_dataset(dialog_data, "daily_dialog_sample.json")
    
    # 4. WikiText sample
    print("\n4️⃣ Downloading WikiText sample...")
    wikitext_data = download_wikitext_sample(100)
    save_dataset(wikitext_data, "wikitext_sample.json")
    
    print("\n" + "="*50)
    print("✅ All datasets downloaded successfully!")
    print("\n📁 Available datasets:")
    for filename in os.listdir("datasets"):
        if filename.endswith(".json"):
            filepath = os.path.join("datasets", filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"   • {filename}: {len(data)} Q&A pairs")
    
    print("\n💡 To use a dataset in your model:")
    print("   1. Copy the JSON content to mini_gpt/config.py")
    print("   2. Replace TRAINING_DATA with your chosen dataset")
    print("   3. Run: python3 -m mini_gpt.main")


if __name__ == "__main__":
    main() 