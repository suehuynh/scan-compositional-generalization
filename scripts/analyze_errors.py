import sys
sys.path.insert(0, '.')

import torch
import json
from pathlib import Path
from models.t5_seq2seq import T5SeqToSeq
from data.dataloader import create_dataloaders
from evaluation.error_analysis import ErrorAnalyzer
from evaluation.failure_patterns import CompositionAnalyzer
from config.config import *

def main():
    print("=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)
    
    # Load model
    print("\n[1] Loading model...")
    model = T5SeqToSeq.load('results/models/best_model', device=DEVICE)
    
    # Load test data
    print("\n[2] Loading test data...")
    for split_name, split_path in COMP_TEST_PATHS.items():
        print(f"\n{'='*50}")
        print(f"ANALYZING {split_name.upper()}")
        print(f"{'='*50}")
        
        # Load data
        _, comp_loader, _ = create_dataloaders(
            train_path=TRAIN_PATH,
            comp_test_path=split_path,
            batch_size=BATCH_SIZE
        )
    
        # Analyze errors
        print(f"\n[3] Analyzing failures for {split_name}...")
        analyzer = ErrorAnalyzer(model)
        
        with torch.no_grad():
            for batch in comp_loader:
                input_ids, output_ids, input_attn, output_attn = batch
                input_ids = input_ids.to(DEVICE)
                output_ids = output_ids.to(DEVICE)
                input_attn = input_attn.to(DEVICE)
                
                # Generate predictions
                preds = model.generate(input_ids, input_attn)
                pred_texts = [model.decode(p) for p in preds]
                
                # Analyze this batch
                analyzer.analyze_batch(input_ids, output_ids, input_attn, output_attn, pred_texts)
        
        # Generate report
        print(f"\n[4] Generating report for {split_name}...")
        report = analyzer.generate_report()
        error_dir = Path('results/error_breakdown')
        error_dir.mkdir(parents=True, exist_ok=True)
        analyzer.save_report(error_dir / f'{split_name}_errors.csv')
    
        # Analyze by phenomena
        print(f"\n[5] Analyzing by phenomena for {split_name}...")
        phenomena_analysis = CompositionAnalyzer.analyze_failures_by_phenomena(analyzer.errors)
        phenomena_path = Path(f'results/failure_patterns/{split_name}_phenomena_analysis.json')
        phenomena_path.parent.mkdir(parents=True, exist_ok=True)

        with open(phenomena_path, 'w') as f:
            json.dump(phenomena_analysis, f, indent=2)

    
        print("\n" + "=" * 50)
        print(f"ERROR ANALYSIS {split_name.upper()} COMPLETE")
        print("=" * 50)
    
        # Print summary
        print(f"\nTotal errors: {len(analyzer.errors)}")
        print(f"\nError breakdown:")
        for error_type, count in report['error_counts'].items():
            print(f"  {error_type}: {count}")
        
        print(f"\nFailures by phenomena:")
        for phenomena, stats in phenomena_analysis.items():
            print(f"  {phenomena}: {stats['count']} failures ({stats['rate']:.2%})")

if __name__ == "__main__":
    main()