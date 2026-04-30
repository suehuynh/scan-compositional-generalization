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
    train_loader, comp_loader, tokenizer = create_dataloaders(
        train_path=TRAIN_PATH,
        comp_test_path=COMP_TEST_PATHS['standard'],
        batch_size=BATCH_SIZE
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(comp_loader)}")
    
    
    # Initialize model
    print("\n[2] Loading T5 model...")
    model = T5SeqToSeq(model_name=MODEL_NAME, device=DEVICE)
    print(f"Model loaded on {DEVICE}")
    
    # Create trainer
    print("\n[3] Setting up trainer...")
    trainer = T5Trainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        save_dir=SAVE_DIR
    )
    
    # Train
    print("\n[4] Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=comp_loader,
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Save history
    trainer.save_history()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()