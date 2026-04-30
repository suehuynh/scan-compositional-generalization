import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path
from models.t5_seq2seq import T5SeqToSeq
from evaluation.error_analysis import ErrorAnalyzer
from evaluation.attention_visualization import AttentionVisualizer
from config.config import DEVICE, TRAIN_PATH, COMP_TEST_PATHS
from data.dataloader import create_dataloaders

def main():
    print("=" * 50)
    print("VISUALIZING ATTENTION ON FAILURES")
    print("=" * 50)
    
    # Load model
    print("\n Loading model...")
    model = T5SeqToSeq.load('results/models/best_model', device=DEVICE)
    print("Model loaded")
    
    # Analyze errors for compositional split
    for split_name, split_path in COMP_TEST_PATHS.items():
        print(f"\n Analyzing {split_name} split...")
        _, comp_loader, _ = create_dataloaders(
            train_path=TRAIN_PATH,
            comp_test_path=split_path,
            batch_size=32
        )
        
        analyzer = ErrorAnalyzer(model)
        
        with torch.no_grad():
            for batch in comp_loader:
                input_ids, output_ids, input_attn, output_attn = batch
                input_ids = input_ids.to(DEVICE)
                output_ids = output_ids.to(DEVICE)
                input_attn = input_attn.to(DEVICE)
                
                preds = model.generate(input_ids, input_attn)
                pred_texts = [model.decode(p) for p in preds]
                
                analyzer.analyze_batch(input_ids, output_ids, input_attn, output_attn, pred_texts)
        
        # Visualize failures
        print(f"\n[3] Creating attention visualizations...")
        save_dir = f'results/attention_heatmaps/{split_name}'
        AttentionVisualizer.visualize_failures(
            model=model,
            errors=analyzer.errors,
            num_examples=10,
            save_dir=save_dir
        )
    
    print("\n" + "=" * 50)
    print("VISUALIZATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()