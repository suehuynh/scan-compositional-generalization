from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '.')
from data.dataset import SCANDataset
from transformers import T5Tokenizer
from config.config import *

def create_dataloaders(train_path, comp_test_path, batch_size=BATCH_SIZE):
    """
    Create train and test DataLoaders.
    
    Args:
        train_path (str): Path to training data
        comp_test_path (str): Path to compositional test data
    Returns:
        Tuple of (train_loader, comp_test_loader, vocab)
    """

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    train_dataset = SCANDataset(train_path, tokenizer=tokenizer)
    comp_dataset = SCANDataset(comp_test_path, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    comp_loader = DataLoader(comp_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, comp_loader, tokenizer

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING DATALOADERS")
    print("=" * 50)
    
    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, comp_loader, tokenizer = create_dataloaders(
        train_path=TRAIN_PATH,
        comp_test_path=COMP_TEST_PATHS['standard'],
        batch_size=4
    )
    print("Dataloaders created")
    
    # Test batch
    print("\n[2] Testing batch...")
    batch = next(iter(train_loader))
    input_ids, output_ids, input_attn, output_attn = batch
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Output IDs shape: {output_ids.shape}")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED.")
    print("=" * 50)