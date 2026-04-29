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
        train_path='data/scan/simple_split/tasks_train_simple.txt',
        comp_test_path='data/scan/add_prim_split/tasks_test_addprim_jump.txt',
        batch_size=BATCH_SIZE
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(comp_loader)}")

    # Add this to train.py after creating dataloaders:
    print("\n[DEBUG] Verifying datasets are different...")
    train_batch = next(iter(train_loader))
    comp_batch = next(iter(comp_loader))

    print(f"Train batch first input: {train_batch[0][0][:5]}")
    print(f"Comp batch first input: {comp_batch[0][0][:5]}")
    print(f"Are they same? {torch.equal(train_batch[0][0], comp_batch[0][0])}")

    # Also check file paths exist
    import os
    print(f"\nTrain file exists? {os.path.exists('data/scan/simple_split/tasks_train_simple.txt')}")
    print(f"Comp file exists? {os.path.exists('data/scan/add_prim_split/tasks_test_addprim_jump.txt')}")
    
    # Initialize model
    print("\n[2] Loading T5 model...")
    model = T5SeqToSeq(model_name=MODEL_NAME, device=DEVICE)
    print(f"Model loaded on {DEVICE}")
    
    # # Create trainer
    # print("\n[3] Setting up trainer...")
    # trainer = T5Trainer(
    #     model=model,
    #     device=DEVICE,
    #     learning_rate=LEARNING_RATE,
    #     save_dir=SAVE_DIR
    # )
    
    # # Train
    # print("\n[4] Starting training...")
    # trainer.fit(
    #     train_loader=train_loader,
    #     val_loader=comp_loader,
    #     epochs=EPOCHS,
    #     early_stopping_patience=EARLY_STOPPING_PATIENCE
    # )
    
    # # Save history
    # trainer.save_history()
    
    # print("\n" + "=" * 50)
    # print("TRAINING COMPLETE!")
    # print("=" * 50)

if __name__ == "__main__":
    main()