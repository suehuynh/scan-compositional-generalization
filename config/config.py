from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scan"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
MAX_LENGTH = 50  # Max command length
VOCAB_SIZE = 100  # Approximate (will compute from data)