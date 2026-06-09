from pathlib import Path

# Definizione del progetto (rispetto posizione di questo file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path del dataset
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.pt"

# Riproducibilità e Dataset Split
FIXED_SEED = 42
TEST_SIZE = 0.15

# Costanti di default
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 50