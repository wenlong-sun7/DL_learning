"""
Main script to train the Mini GPT Language Model
"""

import torch
import os
import sys
from typing import Optional

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT,
    LEARNING_RATE, NUM_EPOCHS, MAX_OUTPUT_TOKENS, DEBUG, SAVE_MODEL,
    get_training_data, list_available_datasets
)
from data_utils import Vocabulary, prepare_data, words_to_tensor, tensor_to_words
from models import Transformer
from training import create_model_and_optimizer, train_model, infer_recursive
from plotting import plot_training_loss, plot_loss_comparison, plot_gradient_norms, create_training_summary_plot


def train_mini_gpt(dataset_name: str, epochs: Optional[int] = None, learning_rate: Optional[float] = None, save_model: bool = True):
    """
    Train the Mini GPT model with specified dataset
    
    Args:
        dataset_name: Name of the dataset file (e.g., 'simple_qa.json')
        epochs: Number of training epochs (default: from config)
        learning_rate: Learning rate (default: from config)
        save_model: Whether to save the trained model (default: True)
    """
    # Use config defaults if not specified
    if epochs is None:
        epochs = NUM_EPOCHS
    if learning_rate is None:
        learning_rate = LEARNING_RATE
    
    # Load training data from specified dataset
    try:
        training_data = get_training_data(dataset_name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        print("💡 Available datasets:")
        datasets = list_available_datasets()
        if datasets:
            for filename, size in datasets:
                print(f"  • {filename} ({size} Q&A pairs)")
        else:
            print("  No datasets found in datasets/ directory")
        return
    
    print(f"🚀 Starting Mini GPT Training")
    print(f"📊 Dataset: {dataset_name}")
    print(f"📈 Training data size: {len(training_data)} Q&A pairs")
    print(f"⚙️  Model config: {EMBED_SIZE}d embeddings, {NUM_LAYERS} layers, {NUM_HEADS} heads")
    print(f"🎯 Training config: {epochs} epochs, lr={learning_rate}")
    print("-" * 50)
    
    # Create output directory for plots
    output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data and vocabulary
    print("📝 Creating vocabulary...")
    vocabulary = Vocabulary(training_data)
    data_words = [k for k, _ in training_data.items()]
    target_words = [v for _, v in training_data.items()]
    
    print(f"📚 Vocabulary size: {vocabulary.get_vocab_size()}")
    print(f"📊 Training samples: {len(data_words)}")
    
    # Create model, optimizer, and loss function
    print("🏗️  Creating model...")
    model, optimizer, criterion = create_model_and_optimizer(vocabulary.get_vocab_size(), device)
    
    # Convert training data to tensors
    print("🔄 Converting data to tensors...")
    data = words_to_tensor(data_words, vocabulary, device=device)
    targets = words_to_tensor(target_words, vocabulary, device=device)
    
    # Train model
    print("🎯 Starting training...")
    training_history = train_model(model, optimizer, criterion, data, targets, device)
    
    # Extract training metrics
    epoch_losses = training_history['epoch_losses']
    step_losses = training_history['step_losses']
    gradient_norms = training_history['gradient_norms']
    
    # Generate plots
    print("📊 Generating training plots...")
    plot_training_loss(epoch_losses, save_path=os.path.join(output_dir, "training_loss.png"), show_plot=False)
    plot_loss_comparison(epoch_losses, step_losses, save_path=os.path.join(output_dir, "loss_comparison.png"), show_plot=False)
    plot_gradient_norms(gradient_norms, save_path=os.path.join(output_dir, "gradient_norms.png"), show_plot=False)
    create_training_summary_plot(epoch_losses, gradient_norms, save_path=os.path.join(output_dir, "training_summary.png"), show_plot=False)
    
    # Save model if requested
    if save_model:
        print("💾 Saving model...")
        os.makedirs("saved_model", exist_ok=True)
        torch.save(model.state_dict(), "saved_model/model.pth")
        torch.save(vocabulary.word_to_ix, "saved_model/vocabulary.pth")
        print("✅ Model saved to saved_model/")
    
    # Test generation
    print("\n🧪 Testing model generation...")
    input_vector = words_to_tensor(data_words, vocabulary, device=device)
    predicted_vector = infer_recursive(model, input_vector, vocabulary, MAX_OUTPUT_TOKENS)
    predicted_words = tensor_to_words(predicted_vector, vocabulary)
    
    for i, test_input in enumerate(data_words):
        if i < 3:  # Show first 3 examples
            expected = target_words[i]
            generated = predicted_words[i]
            print(f"Input: '{test_input}'")
            print(f"Expected: '{expected}'")
            print(f"Generated: '{generated}'")
            print("-" * 30)
    
    print("🎉 Training completed!")
    print(f"📊 Check the '{output_dir}' directory for training plots!")
    
    return model, vocabulary


def list_datasets():
    """List all available datasets"""
    print("📚 Available datasets:")
    datasets = list_available_datasets()
    if datasets:
        for filename, size in datasets:
            print(f"  • {filename} ({size} Q&A pairs)")
    else:
        print("  No datasets found in datasets/ directory")


if __name__ == "__main__":
    # Example usage
    print("🎯 Mini GPT Training")
    print("=" * 50)
    
    # List available datasets
    list_datasets()
    
    # Train with a specific dataset
    train_mini_gpt("daily_dialog_sample.json", epochs=10) 