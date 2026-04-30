from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scan"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
TRAIN_PATH = 'data/scan/simple_split/tasks_train_simple.txt'
COMP_TEST_PATHS = {
    'standard': 'data/scan/simple_split/tasks_test_simple.txt',
    'addprim_jump': 'data/scan/add_prim_split/tasks_test_addprim_jump.txt',
    'fewshot': 'data/scan/few_shot_split/tasks_test_fewshot_num1_rep1.txt',
    'filler_split': 'data/scan/filler_split/tasks_test_filler_num0.txt',
    'length_split': 'data/scan/length_split/tasks_test_length.txt',
    'template_split': 'data/scan/template_split/tasks_test_template_around_right.txt',
}

# Model hyperparameters
MODEL_NAME = 't5-small'
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOPPING_PATIENCE = 3


# Dataset
MAX_LENGTH = 50  # Max command length
MAX_INPUT_LENGTH = 50
MAX_OUTPUT_LENGTH = 50
MAX_GENERATION_LENGTH = 50
VOCAB_SIZE = 100  # Approximate (will compute from data)

import os
SAVE_DIR = os.path.join(os.path.dirname(__file__), '../results/models')