import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src import config as cfg
from pathlib import Path
from scipy.stats import skew
from sklearn.metrics import confusion_matrix, roc_curve

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

def plot_cv_comparison(
    baseline_scores,
    comparison_scores,
    baseline_std,
    comparison_std,
    baseline_fold_scores,
    comparison_fold_scores,
    comparison_label,
    title,
    save_path,
    show: bool = False
):
    """
    Genera un grafico comparativo tra una configurazione baseline ed
    una configurazione sperimentale.

    Il grafico mostra:

    - media della Cross Validation;
    - deviazione standard (error bar);
    - punteggi ottenuti nei singoli fold della Cross Validation per la metrica riportata sull'asse X.

    Parameters
    ----------
    baseline_scores : ndarray
        Media delle metriche della configurazione baseline.

    comparison_scores : ndarray
        Media delle metriche della configurazione sperimentale.

    baseline_std : ndarray
        Deviazione standard delle metriche baseline.

    comparison_std : ndarray
        Deviazione standard delle metriche sperimentali.

    baseline_fold_scores : list[list]
        Lista dei punteggi ottenuti nei singoli fold.

        Esempio:
        [
            f1_scores,
            auc_scores
        ]

    comparison_fold_scores : list[list]
        Analogo alla baseline.

    comparison_label : str
        Nome della configurazione sperimentale.

    title : str
        Titolo del grafico.

    save_path : Path
        Percorso di esportazione.
    """
    metrics = ["Macro F1", "ROC-AUC"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_scores,
        width,
        yerr=baseline_std,
        capsize=5,
        label="Baseline",
        alpha=0.85,
    )
    bars_comparison = ax.bar(
        x + width / 2,
        comparison_scores,
        width,
        yerr=comparison_std,
        capsize=5,
        label=comparison_label,
        alpha=0.85,
    )

    # Disegno dei punti dei singoli fold
    fixed_offsets = np.array([-0.03, 0.00, 0.03])

    for metric_idx in range(len(metrics)):
        baseline_x = (
            np.full(len(baseline_fold_scores[metric_idx]),
                    x[metric_idx] - width / 2)
            + fixed_offsets[:len(baseline_fold_scores[metric_idx])]
        )
        comparison_x = (
            np.full(len(comparison_fold_scores[metric_idx]),
                    x[metric_idx] + width / 2)
            + fixed_offsets[:len(comparison_fold_scores[metric_idx])]
        )
        ax.scatter(
            baseline_x,
            baseline_fold_scores[metric_idx],
            s=55,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
            label="Singoli fold" if metric_idx == 0 else None,
        )
        ax.scatter(
            comparison_x,
            comparison_fold_scores[metric_idx],
            s=55,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )

    # Valori sopra le barre
    for bars in (bars_baseline, bars_comparison):
        for bar in bars:
            height = bar.get_height()

            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylabel("Punteggio medio in Cross Validation")

    ax.set_title(title)

    ax.set_ylim(0.60, 0.85)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close()

def plot_final_evaluation_metrics(
    history_loss: list, 
    y_true: list, 
    y_pred: list, 
    y_score: list, 
    roc_auc: float, 
    save_path: Path, 
    show: bool = False
):
    """
    Genera e salva la dashboard finale di valutazione del modello.
    Include la Curva di Apprendimento, la Matrice di Confusione e la Curva ROC.
    """
    print("\n>> COMPILAZIONE GRAFICI DI VALUTAZIONE FINALI...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Grafico A: Curva di Loss
    ax[0].plot(range(1, len(history_loss) + 1), history_loss, label='Training Loss', color='teal', linewidth=2)
    ax[0].set_title('Curva di Apprendimento (Final Training)')
    ax[0].set_xlabel('Epoca')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True, linestyle='--', alpha=0.6)
    ax[0].legend()

    # Grafico B: Matrice di Confusione
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1], cbar=False,
                xticklabels=['Non-Enzima', 'Enzima'], yticklabels=['Non-Enzima', 'Enzima'])
    ax[1].set_title('Matrice di Confusione sul Test Set')
    ax[1].set_xlabel('Predetto')
    ax[1].set_ylabel('Reale')

    # Grafico C: Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax[2].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax[2].set_xlim([0.0, 1.0])
    ax[2].set_ylim([0.0, 1.05])
    ax[2].set_title('Curva ROC')
    ax[2].set_xlabel('Tasso di Falsi Positivi')
    ax[2].set_ylabel('Tasso di Veri Positivi')
    ax[2].grid(True, linestyle='--', alpha=0.6)
    ax[2].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        
    plt.close()