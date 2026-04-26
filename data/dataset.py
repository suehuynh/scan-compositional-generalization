import torch
from torch.utils.data import Dataset
from collections import Counter
import pickle
from pathlib import Path

class SCANDataset(Dataset):
    """
    PyTorch Dataset for SCAN data.
    Handles tokenization, vocabulary building, and padding.
    """
    
    def __init__(self, filepath, max_input_length=50, max_output_length=50, 
                 vocab=None, build_vocab=False):
        """
        Args:
            filepath (str): Path to SCAN .txt file
            max_input_length (int): Maximum input sequence length
            max_output_length (int): Maximum output sequence length
            vocab (dict): Pre-built vocabulary with keys 'input' and 'output'
            build_vocab (bool): Whether to build vocabulary from this dataset
        """
        self.filepath = filepath
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Store examples as (input_tokens, output_tokens)
        self.examples = []
        
        # Initialize vocabularies
        self.input_vocab = vocab['input'] if vocab else None
        self.output_vocab = vocab['output'] if vocab else None
        
        # Load examples from file
        self._load_examples()
        
        # Build vocabularies if requested
        if build_vocab:
            self._build_vocab()
    
    def _load_examples(self):
        """
        Load and parse SCAN data from file.
        Return list of tuples of (input commands, output actions)
        """
        print(f"Loading examples from {self.filepath}...")
        
        with open(self.filepath, "r") as f:
            for line in f:
                if line.strip(): # Skip empty lines
                    cmd_act = line.strip().split("OUT: ")
                    input_cmd = cmd_act[0].split("IN: ")[1].split()
                    output_actions = cmd_act[1].split()
                    self.examples.append((input_cmd, output_actions))
    
    def _build_vocab(self):
        """
        Build input and output vocabularies from examples.
        """
        print("Building vocabularies...")
        
        input_tokens = []
        output_tokens = []
        for input_token, output_token in self.examples:
            input_tokens.extend(input_token)
            output_tokens.extend(output_token)
        
        input_counter = Counter(input_tokens)
        output_counter = Counter(output_tokens)

        self.input_vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (token, count) in enumerate(input_counter.most_common(), start=2):
            self.input_vocab[token] = i

        self.output_vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (token, count) in enumerate(output_counter.most_common(), start=2):
            self.output_vocab[token] = i
        print(f"Input vocab size: {len(self.input_vocab)}")
        print(f"Output vocab size: {len(self.output_vocab)}")

    def _tokens_to_ids(self, tokens, vocab, max_length):
        """
        Convert token list to padded tensor of IDs.
        
        Args:
            tokens (list): List of string tokens
            vocab (dict): Vocabulary mapping tokens to IDs
            max_length (int): Length to pad to
        
        Returns:
            torch.LongTensor of shape (max_length,)
        """
        ids = []
        for token in tokens[:max_length]:
            if token in vocab:
                id = vocab.get(token, vocab["<UNK>"])
                ids.append(id)
        ids.extend([vocab["<PAD>"]] * (max_length - len(ids)))
        return torch.LongTensor(ids)
    
    def __len__(self):
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get single example.
        
        Returns:
            Tuple of (input_ids, output_ids, input_length, output_length)
            - input_ids: torch.LongTensor of shape (max_input_length,)
            - output_ids: torch.LongTensor of shape (max_output_length,)
            - input_length: int (actual input length before padding)
            - output_length: int (actual output length before padding)
        """
        input_tokens, output_tokens = self.examples[idx]
        input_ids = self._tokens_to_ids(input_tokens, self.input_vocab, self.max_input_length)
        output_ids = self._tokens_to_ids(output_tokens, self.output_vocab, self.max_output_length)
        input_length = len(input_tokens)
        output_length = len(output_tokens)
        
        return input_ids, output_ids, input_length, output_length


    def save_vocab(self, filepath):
        """Save vocabularies to file."""
        vocab = {
            'input': self.input_vocab,
            'output': self.output_vocab
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Saved vocabularies to {filepath}")
    
    @staticmethod
    def load_vocab(filepath):
        """Load vocabularies from file."""
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Loaded vocabularies from {filepath}")
        return vocab
    
    def get_vocab(self):
        """Return current vocabularies."""
        return {
            'input': self.input_vocab,
            'output': self.output_vocab
        }
    
    def vocab_size(self):
        """Return vocabulary sizes."""
        return {
            'input': len(self.input_vocab),
            'output': len(self.output_vocab)
        }
    
    def _ids_to_tokens(self, ids, vocab, input_or_output='input'):
        """
        Convert ID tensor back to tokens (reverse of _tokens_to_ids).
        
        Args:
            ids (torch.LongTensor): Tensor of IDs, shape (sequence_length,)
            vocab (dict): Vocabulary mapping tokens to IDs
            input_or_output (str): 'input' or 'output' (for which reverse vocab to use)
        
        Returns:
            List of tokens (without <PAD> tokens)
        """
        # Create reverse vocabulary (id -> token)
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        # Convert IDs to tokens
        tokens = []
        for id_val in ids:
            id_val = id_val.item()  # Convert tensor to Python int
            if id_val == 0:  # Stop at PAD token
                break
            token = reverse_vocab.get(id_val, '<UNK>')
            tokens.append(token)
        return tokens