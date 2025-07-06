"""
Prediction script for the Mini GPT Language Model
Loads a trained model and makes predictions on new sentences
"""

import torch
import os
from mini_gpt.models import Transformer
from mini_gpt.data_utils import Vocabulary, words_to_tensor, tensor_to_words


def load_model_and_vocabulary(model_dir="saved_model"):
    """
    Load the trained model and vocabulary
    
    Args:
        model_dir: Directory containing saved model files
        
    Returns:
        Tuple of (model, vocabulary, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model checkpoint
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same architecture
    model = Transformer(
        vocab_size=checkpoint['vocab_size'],
        embed_size=checkpoint['embed_size'],
        num_layers=checkpoint['num_layers'],
        head_counts=checkpoint['num_heads']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load vocabulary
    vocab_path = os.path.join(model_dir, "vocabulary.pth")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    vocab_data = torch.load(vocab_path, map_location=device)
    
    # Create vocabulary object
    vocabulary = Vocabulary.__new__(Vocabulary)  # Create instance without calling __init__
    vocabulary.word_to_ix = vocab_data['word_to_ix']
    vocabulary.ix_to_word = vocab_data['ix_to_word']
    vocabulary.vocab_words = vocab_data['vocab_words']
    
    return model, vocabulary, device


def predict_sentence(model, vocabulary, sentence, device, max_tokens=10):
    """
    Predict the response for a given input sentence
    
    Args:
        model: Trained Transformer model
        vocabulary: Vocabulary object
        sentence: Input sentence string
        device: Device to run inference on
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Predicted response string
    """
    model.eval()
    
    # Convert sentence to tensor
    input_tensor = words_to_tensor([sentence], vocabulary, device=device)
    
    # Initialize input for autoregressive generation
    input_vector = input_tensor[0].reshape(1, input_tensor.shape[1])
    predicted_sequence = []
    word_count = 0
    
    with torch.no_grad():
        while True:
            # Generate next token
            output = model(input_vector)
            predicted_token = output[0, :].argmax().item()
            predicted_sequence.append(predicted_token)
            
            # Stop if end token is generated or max length reached
            if (predicted_token == vocabulary.get_end_token_idx() or 
                word_count >= max_tokens):
                break
            
            # Add predicted token to input for next iteration
            new_token_tensor = torch.tensor([predicted_token]).to(device)
            new_token_tensor = new_token_tensor.unsqueeze(0)
            input_vector = torch.cat([input_vector, new_token_tensor], dim=1)
            word_count += 1
    
    # Convert predicted tokens back to words
    predicted_words = []
    for idx in predicted_sequence:
        if idx == vocabulary.get_end_token_idx():
            break
        predicted_words.append(vocabulary.ix_to_word[idx].lower())
    
    return " ".join(predicted_words)


def main():
    """Main prediction function"""
    
    # Check if model exists
    if not os.path.exists("saved_model"):
        print("❌ No saved model found! Please run training first:")
        print("   python3 -m mini_gpt.main")
        return
    
    try:
        # Load model and vocabulary
        print("🔄 Loading trained model...")
        model, vocabulary, device = load_model_and_vocabulary()
        print(f"✅ Model loaded successfully on {device}")
        
        # Test sentence
        test_sentence = "how am i"
        print(f"\n🤖 Testing prediction for: '{test_sentence}'")
        
        # Make prediction
        prediction = predict_sentence(model, vocabulary, test_sentence, device)
        print(f"📝 Model response: '{prediction}'")
        
        # Interactive mode
        print("\n" + "="*50)
        print("INTERACTIVE PREDICTION MODE")
        print("="*50)
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n💬 Enter a sentence: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("❌ Please enter a valid sentence.")
                    continue
                
                # Make prediction
                prediction = predict_sentence(model, vocabulary, user_input, device)
                print(f"🤖 Model response: '{prediction}'")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Try a different sentence or check if all words are in the vocabulary.")
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you've run the training script first.")


if __name__ == "__main__":
    main() 