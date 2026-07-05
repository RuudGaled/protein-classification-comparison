import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src import config as cfg
from pathlib import Path
from scipy.stats import skew

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

def plot_skewed_features_distributions(
    dataset,
    feature_indices,
    save_path: Path,
    show: bool = False
):
    """
    Genera e salva un grafico composto da Box Plot e Istogramma
    per le prime feature con distribuzione fortemente asimmetrica.
    """

    if not feature_indices:
        print("[INFO] Nessuna feature asimmetrica specificata per il plot.")
        return

    # Estrazione di tutte le feature dei nodi e conversione in NumPy
    all_x = torch.cat([g.x for g in dataset], dim=0).cpu().numpy()

    # Selezione delle prime quattro feature asimmetriche
    features_to_plot = feature_indices[:4]
    n_features = len(features_to_plot)

    # Configurazione dinamica della griglia
    rows = 2 if n_features <= 2 else 4
    height_ratios = [1, 4] if rows == 2 else [1, 4, 1, 4]

    # Numero di righe effettive di grafici (2 grafici per riga)
    n_rows_of_plots = (n_features + 1) // 2

    fig = plt.figure(
        figsize=(14, 5 * n_rows_of_plots),
        constrained_layout=True
    )

    gs = fig.add_gridspec(
        rows,
        2,
        height_ratios=height_ratios,
        hspace=0.3,
        wspace=0.2
    )

    for i, col_idx in enumerate(features_to_plot):

        feature = all_x[:, col_idx]

        feature_skew = skew(feature)
        mean = float(np.mean(feature))
        median = float(np.median(feature))

        col = i % 2
        row_box = 0 if i < 2 else 2
        row_hist = 1 if i < 2 else 3

        ax_box = fig.add_subplot(gs[row_box, col])
        ax_hist = fig.add_subplot(gs[row_hist, col], sharex=ax_box)

        # Box Plot
        sns.boxplot(
            x=feature,
            ax=ax_box,
            color="steelblue",
            fliersize=3
        )

        ax_box.set_title(
            f"Feature {col_idx:02d} (Skewness: {feature_skew:.2f})",
            fontweight="bold"
        )

        ax_box.set_xlabel("")
        ax_box.set_yticks([])
        ax_box.tick_params(
            axis="x",
            bottom=False,
            labelbottom=False
        )

        # Istogramma
        sns.histplot(
            feature,
            ax=ax_hist,
            kde=True,
            bins="fd",
            color="steelblue"
        )

        ax_hist.set_xlabel("Valore")
        ax_hist.set_ylabel("Frequenza (Nodi)")

        # Media
        ax_hist.axvline(
            mean,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Media"
        )
        # Mediana
        ax_hist.axvline(
            median,
            color="orange",
            linestyle="--",
            linewidth=1,
            label="Mediana"
        )

        ax_hist.legend(loc="upper right", fontsize=8)

        ax_hist.grid(alpha=0.3)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close()

    print("[INFO] Esportazione grafico delle distribuzioni completata.\n")