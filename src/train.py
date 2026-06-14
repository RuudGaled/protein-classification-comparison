from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src import config as cfg
from src import utils as ut
from src import data_loader as dl

def run_cross_validation(
    dataset,
    model_class,
    n_splits=cfg.N_SPLITS,
    batch_size=cfg.DEFAULT_BATCH_SIZE,
    train_fold_fn=None,
    seed=cfg.FIXED_SEED,
    lr=cfg.DEFAULT_LR,
    dropout_p=0.3,
    excluded_features=None,
):
    """
    Esegue una Stratified K-Fold Cross Validation raccogliendo Macro F1 e ROC-AUC.
    """
    if train_fold_fn is None:
        raise ValueError("[ERRORE] È necessario fornire una funzione di training (train_fold_fn)")

    labels = [g.y.item() for g in dataset]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    f1_scores = []
    auc_scores = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n   --- FOLD {fold_id}/{n_splits} ---")

        # Split dei grafi per il fold corrente
        train_graphs = [dataset[i] for i in train_idx]
        val_graphs = [dataset[i] for i in val_idx]

        # Normalizzazione Z-Score selettiva (Prevenzione Data Leakage)
        train_graphs, val_graphs = dl.normalize_fold_data(
            train_graphs, val_graphs, excluded_features=excluded_features
        )

        # Creazione dei DataLoader nativi PyG
        train_loader, val_loader = dl.create_dataloaders(
            train_graphs, val_graphs, batch_size=batch_size
        )

        # Dimensione delle feature dei nodi 
        # (tutti i grafi hanno lo stesso numero di feature per nodo)
        num_features = dataset[0].x.shape[1]

        # Istanziamo il modello passando i parametri dinamici per la Grid Search
        model = model_class(in_channels=num_features, hidden_channels=64, dropout_p=dropout_p)

        # Alleniamo il modello sul fold (riceve 3 elementi in output)
        y_true, y_pred, y_score = train_fold_fn(
            model, train_loader, val_loader, lr=lr, epochs=cfg.DEFAULT_EPOCHS
        )

        # Calcolo delle metriche per il fold corrente
        fold_f1 = f1_score(y_true, y_pred, average="macro")
        fold_auc = roc_auc_score(y_true, y_score)

        f1_scores.append(fold_f1)
        auc_scores.append(fold_auc)

        print(f"   Risultati Fold {fold_id} | Macro F1: {fold_f1:.4f} | ROC-AUC: {fold_auc:.4f}")

    return f1_scores, auc_scores

def train_single_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = cfg.DEFAULT_LR,
    epochs: int = cfg.DEFAULT_EPOCHS,
    weight_decay: float = 1e-4
):
    """
    Esegue l'addestramento e la validazione di un modello su un singolo fold.
    Soddisfa i requisiti di regolarizzazione (Weight Decay) e stabilità (BCEWithLogitsLoss).

    Returns
    -------
    tuple[list, list]
        y_true: Etichette reali del validation set.
        y_pred: Predizioni binarie (0 o 1) generate dal modello.
    """
    # 1. Funzione di Loss e Ottimizzatore (con L2 Regularization / Weight Decay)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Spostiamo esplicitamente il modello su CPU (coerente con utils.get_device())
    model.to(cfg.DEVICE)

    # Loop delle Epoche
    for epoch in range(1, epochs + 1):
        # --- FASE DI TRAINING ---
        model.train()
        running_loss = 0.0
        total_graphs = 0  # Contatore dinamico per evitare errori di tipo con Pylance
        
        for batch_data in train_loader:
            optimizer.zero_grad()               # Resetta i gradienti del passo precedente
            
            logits = model(batch_data).squeeze(-1) # Output della rete: [batch_size]
            labels = batch_data.y.float()         # Target reale convertito in float per la loss
            
            loss = criterion(logits, labels)    # Calcolo dell'errore
            loss.backward()                     # Retropropagazione del gradiente
            optimizer.step()                    # Aggiornamento dei pesi
            
            running_loss += loss.item() * batch_data.num_graphs
            total_graphs += batch_data.num_graphs  # Incrementiamo con il numero reale di grafi nel batch

        epoch_loss = running_loss / total_graphs
        
        # Stampiamo l'andamento ogni 10 epoche per monitorare senza intasare l'output
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"   [Epoca {epoch:02d}/{epochs:02d}] Training Loss: {epoch_loss:.4f}")

    # --- FASE DI VALUTAZIONE FINALE DEL FOLD ---
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad(): # Disabilita il calcolo dei gradienti per risparmiare memoria e velocizzare
        for batch_data in val_loader:
            logits = model(batch_data).squeeze(-1)
            labels = batch_data.y

            probs = torch.sigmoid(logits)

            # Conversione dei logits in predizioni binarie (Soglia a 0.5)
            preds = (probs >= 0.5).long()
            
            # Accumuliamo i risultati convertendoli in liste Python standard
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs.tolist())

    return y_true, y_pred, y_score