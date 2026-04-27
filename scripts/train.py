import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path
from models.t5_seq2seq import T5SeqToSeq
from training.trainer import T5Trainer
from data.dataset import SCANDataset
from data.dataloader import create_dataloaders
from config.config import *

def main():
    print("=" * 50)
    print("FINE-TUNING T5 ON SCAN")
    print("=" * 50)
    
    # Load dataloaders
    print("\n[1] Loading data...")
    train_loader, comp_loader, vocab = create_dataloaders(
        train_path='data/scan/simple_split/tasks_train_simple.txt',
        comp_test_path='data/scan/add_prim_split/tasks_test_addprim_jump.txt',
        vocab_path='data/scan/vocab.pkl',
        batch_size=BATCH_SIZE
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(comp_loader)}")
    
    # Initialize model
    print("\n[2] Loading T5 model...")
    model = T5SeqToSeq(model_name='t5-small', device=DEVICE)
    print(f"Model loaded on {DEVICE}")
    
    # Create trainer
    print("\n[3] Setting up trainer...")
    trainer = T5Trainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        save_dir='results/models'
    )
    
    # Train
    print("\n[4] Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=comp_loader,
        epochs=20,
        early_stopping_patience=3
    )
    
    # Save final model
    print("\n[5] Saving model...")
    model.save('results/models/final_model')
    trainer.save_history()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()