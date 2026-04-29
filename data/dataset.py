import torch
from torch.utils.data import Dataset
import sys
sys.path.insert(0, '.')
from collections import Counter
from pathlib import Path
from config.config import *

class SCANDataset(Dataset):
    """
    PyTorch Dataset for SCAN data.
    Handles tokenization, vocabulary building, and padding.
    """
    
    def __init__(self, filepath, tokenizer, max_input_length=MAX_INPUT_LENGTH, max_output_length=MAX_OUTPUT_LENGTH, 
                 vocab=None, build_vocab=False):
        self.filepath = filepath
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer 
        self.examples = []
        self._load_examples()
    
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
    
    def __len__(self):
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_tokens, output_tokens = self.examples[idx]
        input_ids = self._tokens_to_ids(input_tokens, self.input_vocab, self.max_input_length)
        output_ids = self._tokens_to_ids(output_tokens, self.output_vocab, self.max_output_length)

        input_attention_mask = (input_ids != 0).long()
        output_attention_mask = (output_ids != 0).long()
        
        return input_ids, output_ids, input_attention_mask, output_attention_mask

    def __getitem__(self, idx):
        input_tokens, output_tokens = self.examples[idx]
        
        input_text = ' '.join(input_tokens)
        output_text = ' '.join(output_tokens)
        
        input_ids = self.tokenizer(input_text, max_length=50, padding='max_length', 
                                truncation=True, return_tensors='pt')['input_ids'].squeeze()
        output_ids = self.tokenizer(output_text, max_length=50, padding='max_length',
                                truncation=True, return_tensors='pt')['input_ids'].squeeze()
        
        input_attn = (input_ids != 0).long()
        output_attn = (output_ids != 0).long()
        
        return input_ids, output_ids, input_attn, output_attn
    
    def _ids_to_tokens(self, token_ids):
        """Decode token IDs back to tokens using T5 tokenizer."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)