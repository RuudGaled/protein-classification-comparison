from pathlib import Path

# Directory radice del progetto (rispetto posizione di questo file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path del dataset
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.pt"

# Path directory risultati
RESULTS_DIR = PROJECT_ROOT / "results"

# Seed per la riproducibilità degli esperimenti.
FIXED_SEED = 42

# Percentuale del dataset destinata al test.
TEST_SIZE = 0.15

# Batch size di default.
DEFAULT_BATCH_SIZE = 32

# Learning rate di default.
DEFAULT_LR = 1e-3

# Numero di epoche di default.
DEFAULT_EPOCHS = 50

# Dispositivo di esecuzione del modello.
DEVICE = "cpu"

# Numero di fold utilizzati nella Cross-Validation.
N_SPLITS = 3

# Valore del rumore (Noise Injection)
NOISE_LEVEL = 0.01

# Canali di default usati nella griglia principale
DEFAULT_HIDDEN_CHANNELS = 64       
# Canali alternativi testati nel test mirato
ALT_HIDDEN_CHANNELS = 32