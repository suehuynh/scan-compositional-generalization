import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path
from models.t5_seq2seq import T5SeqToSeq
from data.dataloader import create_dataloaders
from data.dataset import SCANDataset
from config.config import *

def exact_match_accuracy(predictions, targets):
    """
    Calculate exact match accuracy.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
    
    Returns:
        Accuracy (0-1)
    """
    n_corrects = sum(1 for p, t in zip(predictions,  targets) if p == t)
    return n_corrects / len(predictions)

def evaluate_model(model, test_loader, tokenizer, max_length=MAX_GENERATION_LENGTH):
    """
    Evaluate model on test set.
    
    Args:
        model: T5SeqToSeq instance
        test_loader: DataLoader
        tokenizer: T5Tokenizer
        max_length: Max generation length
    
    Returns:
        Dictionary with results
    """
    model.eval()
    tokenizer = model.tokenizer

    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, output_ids, input_attn, output_attn = batch
            # Move to device
            input_ids = input_ids.to(DEVICE)
            output_ids = output_ids.to(DEVICE)
            input_attn = input_attn.to(DEVICE)
            # Predictions
            preds = model.generate(input_ids, input_attn, max_length=max_length)

            for pred, target in zip(preds, output_ids):
                # Decode
                pred_tokens = model.decode(pred)
                target_tokens = model.decode(target)
                
                all_predictions.append(pred_tokens)
                all_targets.append(target_tokens)
    accuracy = exact_match_accuracy(all_predictions, all_targets)
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets
    }

if __name__ == "__main__":
    # Load trained model
    model_path = 'results/models/best_model'
    model = T5SeqToSeq.load(model_path, device=DEVICE)
    
    # Load test dataloaders
    _, comp_loader, vocab = create_dataloaders(
        train_path='data/scan/simple_split/tasks_train_simple.txt',
        comp_test_path='data/scan/simple_split/tasks_test_simple.txt',
        batch_size=BATCH_SIZE
    )

    # Evaluate
    results = evaluate_model(model, comp_loader, model.tokenizer)
    
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Compositional Accuracy: {results['accuracy']:.2%}")
    print(f"Sample Predictions:")
    for pred, target in zip(results['predictions'][:5], results['targets'][:5]):
        print(f"  Pred:   {pred}")
        print(f"  Target: {target}")