"""
Training and inference functions for the Mini GPT Language Model
"""

import torch
import torch.nn as nn
from data_utils import pad_tensors


def train_recursive(model, data, targets, optimizer, criterion):
    """
    Train the model using recursive (autoregressive) training
    
    Args:
        model: Transformer model
        data: Input sequences tensor
        targets: Target sequences tensor
        optimizer: PyTorch optimizer
        criterion: Loss function
        
    Returns:
        Tuple of (average_loss, step_losses, gradient_norm)
    """
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    num_steps = 0
    step_losses = []

    batch_size, token_count, token_count_out = data.shape[0], data.shape[1], targets.shape[1]
    
    # Debug info
    from config import DEBUG
    if DEBUG:
        print(f"  Batch size: {batch_size}, Input tokens: {token_count}, Target tokens: {token_count_out}")
        print(f"  Data shape: {data.shape}, Targets shape: {targets.shape}")
    
    for b in range(batch_size):
        end_encountered = False
        cur_count = 0
        
        while not end_encountered:
            # Create target vector for current step
            target_vector = torch.zeros(model.vocab_size).to(data.device)
            if cur_count < token_count_out:
                expected_next_token_idx = targets[b, cur_count]
                target_vector[expected_next_token_idx] = 1

            # Prepare model input
            if cur_count > 0:
                model_input = data[b].reshape(token_count).to(data.device)
                part_of_output = targets[b, :cur_count].to(data.device)
                model_input = torch.cat((model_input, part_of_output))
            else:
                model_input = data[b]

            # Forward pass
            out = model(model_input.reshape(1, token_count + cur_count))
            loss = criterion(out, target_vector.reshape(out.shape))
            
            # Debug info
            if DEBUG and b == 0 and cur_count < 3:  # Only print for first batch, first few steps
                print(f"    Step {cur_count}: Loss = {loss.item():.4f}, Expected token = {expected_next_token_idx}")
                print(f"    Output shape: {out.shape}, Target shape: {target_vector.shape}")
            
            # Accumulate loss
            total_loss += loss.item()
            step_losses.append(loss.item())
            num_steps += 1
            
            # Backward pass for each step (important for gradient flow)
            loss.backward()
            
            cur_count += 1
            if cur_count >= token_count_out:
                end_encountered = True
    
    # Check gradients before update
    gradient_norm = 0
    if DEBUG:
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norm = total_norm ** (1. / 2)
        print(f"  Gradient norm before update: {gradient_norm:.6f}")
    
    # Update weights
    optimizer.step()
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss, step_losses, gradient_norm


def infer_recursive(model, input_vectors, vocabulary, max_output_token_count=10):
    """
    Perform recursive inference with the trained model
    
    Args:
        model: Trained Transformer model
        input_vectors: Input sequences tensor
        vocabulary: Vocabulary object
        max_output_token_count: Maximum number of tokens to generate
        
    Returns:
        Tensor of predicted sequences
    """
    model.eval()
    outputs = []

    for i in range(input_vectors.shape[0]):
        print(f"Inferring sequence {i+1}/{input_vectors.shape[0]}")
        
        input_vector = input_vectors[i].reshape(1, input_vectors.shape[1])
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
                    word_count > max_output_token_count):
                    break
                
                # Add predicted token to input for next iteration
                new_token_tensor = torch.tensor([predicted_token]).to(input_vector.device)
                new_token_tensor = new_token_tensor.unsqueeze(0)
                input_vector = torch.cat([input_vector, new_token_tensor], dim=1)
                word_count += 1

        outputs.append(torch.tensor(predicted_sequence))
    
    return pad_tensors(outputs)


def create_model_and_optimizer(vocab_size, device):
    """
    Create model, optimizer, and loss function
    
    Args:
        vocab_size: Size of vocabulary
        device: Device to place model on
        
    Returns:
        Tuple of (model, optimizer, criterion)
    """
    from config import EMBED_SIZE, NUM_LAYERS, NUM_HEADS, LEARNING_RATE
    from models import Transformer
    
    # Create model
    model = Transformer(vocab_size, EMBED_SIZE, NUM_LAYERS, NUM_HEADS).to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def train_model(model, optimizer, criterion, data, targets, device):
    """
    Train the model for specified number of epochs
    
    Args:
        model: Transformer model
        optimizer: PyTorch optimizer
        criterion: Loss function
        data: Training data tensor
        targets: Target data tensor
        device: Device to train on
        
    Returns:
        Dictionary containing training history
    """
    from config import NUM_EPOCHS
    
    epoch_losses = []
    all_step_losses = []
    gradient_norms = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        loss, step_losses, grad_norm = train_recursive(model, data, targets, optimizer, criterion)
        epoch_losses.append(loss)
        all_step_losses.extend(step_losses)
        gradient_norms.append(grad_norm)
        print(f"Loss: {loss:.4f}")
        
        # Early stopping if loss is not improving
        if epoch > 5 and abs(epoch_losses[-1] - epoch_losses[-2]) < 0.001:
            print("Loss not improving significantly, stopping early.")
            break
    
    return {
        'epoch_losses': epoch_losses,
        'step_losses': all_step_losses,
        'gradient_norms': gradient_norms
    } 