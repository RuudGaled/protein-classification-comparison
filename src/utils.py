import os
import random
import numpy as np
import torch
from src import config as cfg

# Definizione funz. setup_reproducibility
def setup_reproducibility(seed: int = cfg.FIXED_SEED):
    """
    Configura tutti i seed del sistema per garantire la totale 
    riproducibilità dei risultati su CPU.
    """
    # Seed standard
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Controllo Multi-threading (Evita micro-oscillazioni nei float)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    # Forzatura determinismo per CPU impostando i warning invece del blocco totale
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"[INFO] Configurazione di riproducibilità completata con Seed: {seed}")

def get_device() -> torch.device:
    """
    Ritorna esplicitamente il device CPU per preservare la consistenza 
    matematica dei risultati richiesti dal progetto.
    """
    return torch.device(cfg.DEVICE)

def summarize_cv_results(scores):
    """
    Calcola media e deviazione standard
    dei punteggi ottenuti nei fold.
    """

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print("\n===== CROSS VALIDATION =====")

    print(
        f"Mean Macro F1 : {mean_score:.4f}"
    )

    print(
        f"Std Macro F1  : {std_score:.4f}"
    )

    print("============================\n")

    return mean_score, std_score