import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')
from config.config import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5SeqToSeq(nn.Module):
    """
    T5 wrapper for SCAN task.
    
    T5 format:
    - Input: "translate English to SCAN: turn left" 
    - Output: "TURN_LEFT"
    """
    
    def __init__(self, model_name, device):
        """
        Args:
            model_name (str): Pretrained T5 model ('t5-small', 't5-base', etc.)
            device (str): 'cpu' or 'cuda'
        """
        super(T5SeqToSeq, self).__init__()
        self.device = device
        self.model_name = model_name

        # Load pretrained T5
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for training/inference.
        
        Args:
            input_ids: torch.LongTensor, shape (batch_size, input_length)
            attention_mask: torch.LongTensor, shape (batch_size, input_length)
            labels: torch.LongTensor, shape (batch_size, output_length) [optional, for training]
        
        Returns:
            If labels provided: loss (for training)
            Else: logits (for inference)
        """
        if labels is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs # Return loss and logits if in training model
        else:
            output_ids = self.generate(input_ids, attention_mask)
            return output_ids # Return inference if not in training mode
    
    def encode(self, text, max_length=MAX_INPUT_LENGTH):
        """Tokenize input text."""
        encoded = self.tokenizer(text,
                                max_length=max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        return input_ids, attention_mask
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to text."""
        if token_ids.dim() == 1:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        
    def generate(self, input_ids, attention_mask, max_length=MAX_GENERATION_LENGTH):
        """Generate output sequences."""
        output_ids = self.model.generate(
                                    input_ids=input_ids, 
                                    attention_mask=attention_mask,
                                    max_length=max_length,
                                    num_beams=1) # Greedy decoding
        return output_ids

    def train(self):
        """Set model to training mode."""
        self.model.train()
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
        
    def save(self, filepath):
        """Save model checkpoint."""
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
    
    @staticmethod
    def load(filepath, model_name=MODEL_NAME, device=DEVICE):
        """Load model from checkpoint."""
        model = T5SeqToSeq(model_name, device)
        model.model = T5ForConditionalGeneration.from_pretrained(filepath).to(device)
        model.tokenizer = T5Tokenizer.from_pretrained(filepath)
        return model

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING T5SeqToSeq")
    print("=" * 50)

    # Load model
    print("\n[1] Loading T5 model...")
    model = T5SeqToSeq(model_name=MODEL_NAME, device=DEVICE)
    print(f"Model loaded on {DEVICE}")

    # Encode
    print("\n[2] Testing encode()...")
    text = "translate English to actions: turn left"
    input_ids, attn_mask = model.encode(text)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attn_mask.shape}")

    # Generate (inference)
    print("\n[3] Testing generate()...")
    output_ids = model.generate(input_ids, attn_mask)
    print(f"Output IDs shape: {output_ids.shape}")

    # Test 4: Decode
    print("\n[4] Testing decode()...")
    output_text = model.decode(output_ids[0])
    print(f"Decoded output: {output_text}")

    # Test 5: Forward pass (training mode with fake labels)
    print("\n[5] Testing forward() with labels...")
    test_labels = torch.tensor([[101, 2054, 2054, 0, 0, 0]]).to(DEVICE)
    outputs = model.forward(input_ids, attn_mask, labels=test_labels)
    print(f"Loss: {outputs.loss.item():.4f}")

    # Test 6: Forward pass (inference mode without labels)
    print("\n[6] Testing forward() without labels...")
    output_ids = model.forward(input_ids, attn_mask)
    print(f"Output IDs shape: {output_ids.shape}")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)