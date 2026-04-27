from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '.')
from data.dataset import SCANDataset


def create_dataloaders(train_path, comp_test_path, vocab_path, 
                       batch_size=32, max_input_len=50, max_output_len=50):
    """
    Create train and test DataLoaders.
    
    Args:
        train_path (str): Path to training data
        comp_test_path (str): Path to compositional test data
        vocab_path (str): Path to saved vocabulary
        batch_size (int): Batch size
        max_input_len (int): Max input length
        max_output_len (int): Max output length
    
    Returns:
        Tuple of (train_loader, comp_test_loader, vocab)
    """
    
    # Build train dataset and vocabulary
    train_dataset = SCANDataset(
        filepath=train_path,
        max_input_length=max_input_len,
        max_output_length=max_output_len,
        build_vocab=True
    )
    train_dataset.save_vocab(vocab_path)
    vocab = train_dataset.get_vocab()
    
    # Load test dataset with same vocabulary
    comp_test_dataset = SCANDataset(
        filepath=comp_test_path,
        max_input_length=max_input_len,
        max_output_length=max_output_len,
        vocab=vocab,
        build_vocab=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    comp_test_loader = DataLoader(
        comp_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, comp_test_loader, vocab

if __name__ == "__main__":
    print("=" * 50)
    print("DATALOADERS UNIT TESTING")
    print("=" * 50)
    
    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, comp_test_loader, vocab = create_dataloaders(
        train_path='data/scan/simple_split/tasks_train_simple.txt',
        comp_test_path='data/scan/add_prim_split/tasks_test_addprim_jump.txt',
        vocab_path='data/scan/vocab.pkl',
        batch_size=32,
        max_input_len=50,
        max_output_len=50
    )
    print("Dataloaders created")
    
    # Test train loader
    print("\n[2] Testing train DataLoader...")
    batch = next(iter(train_loader))
    print(f"Train batch input shape: {batch[0].shape}")
    print(f"Train batch output shape: {batch[1].shape}")
    
    # Test comp loader
    print("\n[3] Testing compositional test DataLoader...")
    batch = next(iter(comp_test_loader))
    print(f"Comp batch input shape: {batch[0].shape}")
    print(f"Comp batch output shape: {batch[1].shape}")
    
    # Test decoding
    print("\n[4] Testing token decoding...")
    train_dataset = next(iter(train_loader.dataset.__class__(
        filepath='data/scan/simple_split/tasks_train_simple.txt',
        vocab=vocab,
        build_vocab=False
    )))
    
    print("\n" + "=" * 50)
    print("DATALOADER PASSED!")
    print("=" * 50)