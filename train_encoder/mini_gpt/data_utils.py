"""
Data utilities for the Mini GPT Language Model
Handles vocabulary creation, data preprocessing, and tensor conversions
"""

import torch
from config import END_TOKEN, PAD_TOKEN


class Vocabulary:
    """Manages vocabulary and word-to-index mappings"""
    
    def __init__(self, training_data):
        self.training_data = training_data
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.vocab_words = []
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build vocabulary from training data"""
        data_words = [k for k, _ in self.training_data.items()]
        target_words = [v for _, v in self.training_data.items()]
        
        # Extract all unique words
        all_words = []
        for words in data_words + target_words:
            all_words.extend(words.lower().split(" "))
        
        # Create vocabulary with special tokens
        self.vocab_words = list(set(all_words))
        if END_TOKEN in self.vocab_words:
            self.vocab_words.remove(END_TOKEN)
        self.vocab_words.append(END_TOKEN)
        self.vocab_words.insert(0, PAD_TOKEN)
        
        # Create mappings
        self.word_to_ix = {word.lower(): idx for idx, word in enumerate(self.vocab_words)}
        self.ix_to_word = {idx: word for word, idx in self.word_to_ix.items()}
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return len(self.vocab_words)
    
    def get_end_token_idx(self):
        """Get index of end token"""
        return self.word_to_ix[END_TOKEN]


def pad_tensors(tensor_list):
    """
    Pad a list of tensors to the same length
    
    Args:
        tensor_list: List of tensors to pad
        
    Returns:
        Padded tensor with shape (batch_size, max_len, ...)
    """
    if not tensor_list:
        return torch.tensor([])
    
    max_len = max(tensor.shape[0] for tensor in tensor_list)
    padded_tensors = []
    
    for tensor in tensor_list:
        padded_tensor = torch.zeros(max_len, *tensor.shape[1:], 
                                  dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:tensor.shape[0]] = tensor
        padded_tensors.append(padded_tensor)
    
    return torch.stack(padded_tensors)


def words_to_tensor(seq_batch, vocabulary, device=None):
    """
    Convert a batch of word sequences to tensor indices
    
    Args:
        seq_batch: List of word sequences
        vocabulary: Vocabulary object
        device: Device to place tensors on
        
    Returns:
        Tensor of shape (batch_size, max_seq_len)
    """
    index_batch = []
    
    for seq in seq_batch:
        words_list = seq.lower().split(" ")
        index_list = [vocabulary.word_to_ix[word.lower()] 
                     for word in words_list if word.lower() in vocabulary.word_to_ix]
        tensor = torch.tensor(index_list, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        index_batch.append(tensor)
    
    return pad_tensors(index_batch)


def tensor_to_words(tensor, vocabulary):
    """
    Convert tensor indices back to words
    
    Args:
        tensor: Tensor of indices
        vocabulary: Vocabulary object
        
    Returns:
        List of word sequences
    """
    id_batch = tensor.cpu().numpy().tolist()
    result = []
    
    for indices in id_batch:
        words = []
        for idx in indices:
            if idx == vocabulary.get_end_token_idx():
                break
            words.append(vocabulary.ix_to_word[idx].lower())
        result.append(" ".join(words))
    
    return result


def prepare_data(training_data):
    """
    Prepare training data and vocabulary
    
    Args:
        training_data: Dictionary of training data
        
    Returns:
        Tuple of (vocabulary, data_words, target_words)
    """
    vocabulary = Vocabulary(training_data)
    data_words = [k for k, _ in training_data.items()]
    target_words = [v for _, v in training_data.items()]
    
    return vocabulary, data_words, target_words 