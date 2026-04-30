import torch
import pandas as pd
from collections import defaultdict

class ErrorAnalyzer:
    """Analyze and categorize model failures."""
    
    def __init__(self, model):
        self.model = model
        self.errors = []
    
    def analyze_batch(self, input_ids, output_ids, input_attn, output_attn, predictions):
        """
        Analyze predictions in a batch.
        
        Args:
            input_ids: Tensor, shape (batch_size, max_length)
            output_ids: Tensor, shape (batch_size, max_length)
            input_attn: Tensor
            output_attn: Tensor
            predictions: List of predicted strings
        """
        for i in range(len(input_ids)):
            input_text = self.model.decode(input_ids[i])
            target_text = self.model.decode(output_ids[i])
            pred_text = predictions[i]

            if pred_text != target_text:
                category = self.categorize_error(input_text, target_text, pred_text)
                
                self.errors.append({
                    'input': input_text,
                    'target': target_text,
                    'prediction': pred_text,
                    'error_type': category
                })
    
    def categorize_error(self, input_text, target_text, pred_text):
        """
        Categorize what type of error occurred.
        
        Error types:
        - INCOMPLETE: Missing tokens at end
        - WRONG_ACTION: Wrong action generated
        - WRONG_ORDER: Correct actions but wrong order
        - REPETITION: Too many repetitions
        - TRUNCATED: Output cut off early
        - OTHER: Doesn't fit other categories
        """
        target_tokens = target_text.split()
        pred_tokens = pred_text.split()

        # if prediction is short
        if len(pred_tokens) < len(target_tokens):
            return 'INCOMPLETE'

        # if prediction is long
        if len(pred_tokens) > len(target_tokens):
            # if mismatched actions are wrong
            if any(t not in pred_tokens for t in target_tokens):
                return 'WRONG ACTION'
            return 'REPETITION'
        
        # if  are right but wrong order
        if set(pred_tokens) == set(target_tokens):
            return 'WRONG ORDER'
        
        # Check if some actions are wrong
        if any(t not in pred_tokens for t in target_tokens):
            return 'WRONG ACTION'
        
        return 'OTHER'

    def generate_report(self):
        """
        Generate error analysis report.
        """
        error_df = pd.DataFrame(self.errors)
        error_counts = error_df['error_type'].value_counts().to_dict()

        examples_per_type = {}
        for error_type in error_counts.keys():
            subset = error_df[error_df['error_type'] == error_type].head(3)
            examples_per_type[error_type] = subset.to_dict('records')
        
        return {
            'error_counts': error_counts,
            'examples_per_type': examples_per_type,
            'total_errors': len(self.errors)
        }
    
    def save_report(self, filepath):
        """Save error report to CSV."""
        df = pd.DataFrame(self.errors)
        df.to_csv(filepath, index=False)
        print(f"Error report saved to {filepath}")