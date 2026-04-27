import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')
from config.config import *
class T5Trainer:
    """
    Trainer for T5 seq2seq fine-tuning on SCAN.
    """
    
    def __init__(self, model, device=DEVICE, learning_rate=LEARNING_RATE, save_dir=SAVE_DIR):
        """
        Args:
            model: T5SeqToSeq instance
            device: 'cpu' or 'cuda'
            learning_rate: float
            save_dir: where to save checkpoints
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader with batches
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss= 0
        for batch in train_loader:
            input_ids, output_ids, input_attn, output_attn = batch
            input_ids = input_ids.to(self.device)
            output_ids = output_ids.to(self.device)
            input_attn = input_attn.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, input_attn, labels=output_ids)
            loss = outputs.loss

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
        return avg_loss
        
    
    def eval_epoch(self, val_loader):
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: DataLoader
        
        Returns:
            Average loss, accuracy
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids, output_ids, input_attn, output_attn = batch

                input_ids = input_ids.to(self.device)
                output_ids = output_ids.to(self.device)
                input_attn = input_attn.to(self.device)

                outputs = self.model(input_ids, input_attn, labels=output_ids)
                loss = outputs.loss
                total_loss += loss.item()

                predictions = self.model.generate(input_ids, input_attn)

                for pred, target, target_len in zip(predictions, output_ids, output_lengths):
                    pred_tokens = self.model.decode(pred)
                    target_tokens = self.model.decode(target[:target_len])

                    if pred_tokens == target_tokens:
                        total_correct += 1
                total_samples += 1
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_accuracy
    
    def fit(self, train_loader, val_loader, epochs=10, early_stopping_patience=3):
        """
        Train for multiple epochs with early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1} / {epochs}")
            print(f"\n{'='*50}")

            avg_train_loss = self.train_epoch(train_loader)
            avg_val_loss, avg_val_accuracy = self.eval_epoch(val_loader)

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_accuracy'].append(avg_val_accuracy)
            
            print(f"Train loss: {avg_train_loss:.4f}")
            print(f"Val loss: {avg_val_loss:.4f}")
            print(f"Val accuracy: {avg_val_accuracy:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_best_model(epoch, avg_val_loss)
                print(f"Best model saved!")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{early_stopping_patience})")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*50}")
        print(f"Training complete!")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"{'='*50}")
        
        self.save_history()
    
    def save_checkpoint(self, epoch, loss, filepath=None):
        """Save model checkpoint."""
        if filepath is None:
            filepath = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
    
    def _save_best_model(self, epoch, loss):
        """Save best model checkpoint."""
        checkpoint_path = self.save_dir / f'best_model_epoch_{epoch}.pt'
        self.model.save(checkpoint_path)
        
        # Save as 'best_model' (latest best)
        best_path = self.save_dir / 'best_model'
        self.model.save(best_path)
    
    def save_history(self, filepath=None):
        """Save training history to JSON."""
        if filepath is None:
            filepath = self.save_dir / 'training_history.json'
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {filepath}")