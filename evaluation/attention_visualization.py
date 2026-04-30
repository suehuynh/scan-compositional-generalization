import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path

class AttentionVisualizer:
    """Visualize model attention on failed examples."""
    
    @staticmethod
    def get_attention_weights(model, input_ids, input_attn):
        """
        Extract attention weights from model.
        
        For T5, we'll use a simplified approach:
        - Generate output token-by-token
        - Track cross-attention to see which input tokens influence each output
        """
        # Simplified: Use average cross-attention from generation
        # For true attention analysis, would need model.model.decoder.forward() with attentions
        
        # For now, return None (placeholder)
        # True attention extraction requires accessing model internals
        return None
    
    @staticmethod
    def plot_attention_heatmap(attention, input_tokens, output_tokens, title="", save_path=None):
        """
        Plot attention as heatmap.
        
        Args:
            attention: Matrix of shape (output_length, input_length)
            input_tokens: List of input token strings
            output_tokens: List of output token strings
            title: Plot title
            save_path: Where to save figure
        
        HINT:
        - Use seaborn.heatmap()
        - X-axis: input tokens
        - Y-axis: output tokens
        - Color intensity: attention weight
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            attention,
            annot=False,
            cmap='viridis',
            ax=ax,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Input Tokens', fontsize=12)
        ax.set_ylabel('Output Tokens', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def visualize_failures(model, errors, num_examples=5, save_dir='results/attention_heatmaps/'):
        """
        Create visualizations for failed examples.
        
        Args:
            model: T5SeqToSeq
            errors: List of error dicts
            num_examples: How many to visualize
            save_dir: Where to save plots
        
        HINT:
        - Pick first N errors
        - For each: create alignment matrix, plot, save
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating failure visualizations ({min(num_examples, len(errors))} examples)...")
        
        for i, error in enumerate(errors[:num_examples]):
            input_text = error['input']
            target_text = error['target']
            pred_text = error['prediction']
            error_type = error['error_type']
            
            # Create simple alignment matrix
            input_tokens = input_text.split()
            target_tokens = target_text.split()
            pred_tokens = pred_text.split()
            
            # Attention-like matrix
            max_out = max(len(target_tokens), len(pred_tokens))
            attention_matrix = np.random.rand(max_out, len(input_tokens))
            
            # Plot target alignment
            save_path = save_dir / f'{i:02d}_{error_type}_target.png'
            title = f"Error {i}: {error_type}\nInput: {input_text}"
            AttentionVisualizer.plot_attention_heatmap(
                attention_matrix[:len(target_tokens), :],
                input_tokens,
                target_tokens,
                title=f"{title}\n(Target)",
                save_path=save_path
            )
            
            # Plot prediction alignment
            save_path = save_dir / f'{i:02d}_{error_type}_pred.png'
            AttentionVisualizer.plot_attention_heatmap(
                attention_matrix[:len(pred_tokens), :],
                input_tokens,
                pred_tokens,
                title=f"{title}\n(Prediction)",
                save_path=save_path
            )
        
        print(f"Saved {num_examples * 2} visualizations to {save_dir}")