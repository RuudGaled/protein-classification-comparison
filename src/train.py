from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

from src import config as cfg
from src import utils as ut
from src import data_loader as dl

def run_cross_validation(
    dataset,
    model_class,
    n_splits,
    batch_size,
    train_fold_fn,
    seed
):
    """
    Esegue una Stratified K-Fold Cross Validation.

    Parameters
    ----------
    model_class
        Classe del modello GNN.

    train_fold_fn
        Funzione che esegue training e validation
        di un singolo fold.
    """

    labels = [g.y.item() for g in dataset]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    fold_scores = []

    for fold_id, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1
    ):

        print(
            f"\n===== FOLD {fold_id}/{n_splits} ====="
        )

        train_graphs = [
            dataset[i]
            for i in train_idx
        ]

        val_graphs = [
            dataset[i]
            for i in val_idx
        ]

        train_graphs, val_graphs = (
            dl.normalize_fold_data(
                train_graphs,
                val_graphs
            )
        )

        train_loader, val_loader = (
            dl.create_dataloaders(
                train_graphs,
                val_graphs,
                batch_size
            )
        )

        model = model_class()

        y_true, y_pred = train_fold_fn(
            model,
            train_loader,
            val_loader
        )

        fold_f1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        )

        fold_scores.append(fold_f1)

        print(
            f"Fold {fold_id} "
            f"Macro F1 = {fold_f1:.4f}"
        )

    return fold_scores