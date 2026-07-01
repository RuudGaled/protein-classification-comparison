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
    hidden_channels=cfg.DEFAULT_HIDDEN_CHANNELS,
    excluded_features=None,
    noise_level=cfg.NOISE_LEVEL,
):
    """
    Esegue una Stratified K-Fold Cross Validation applicando Z-Score e Noise Injection in modo isolato all'interno di ciascun fold. Restituisce Macro F1 e ROC-AUC.
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

        # Applicazione Noise Injection solo sul train set del fold
        if noise_level > 0.0:
            train_graphs = dl.inject_node_noise(
                train_graphs, 
                noise_level=noise_level, 
                excluded_features=excluded_features
            )

        # Creazione dei DataLoader nativi PyG
        train_loader, val_loader = dl.create_dataloaders(
            train_graphs, val_graphs, batch_size=batch_size
        )

        # Dimensione delle feature dei nodi 
        # (tutti i grafi hanno lo stesso numero di feature per nodo)
        num_features = dataset[0].x.shape[1]

        # Istanziamo il modello passando i parametri dinamici per la Grid Search
        model = model_class(in_channels=num_features, 
                            hidden_channels=hidden_channels, 
                            dropout_p=dropout_p)

        # Training sul singolo fold
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

def evaluate_model(model: nn.Module, data_loader: DataLoader):
    """
    Esegue l'inferenza su un DataLoader estraendo le etichette reali, 
    le predizioni binarie e le probabilità (score).
    """
    model.eval()
    model.to(cfg.DEVICE)
    
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(cfg.DEVICE)

            logits = model(batch_data).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            y_true.extend(batch_data.y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs.cpu().tolist()) 

    return y_true, y_pred, y_score

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

    # Spostiamo esplicitamente il modello su CPU 
    model.to(cfg.DEVICE)

    # Loop delle Epoche
    for epoch in range(1, epochs + 1):
        # --- FASE DI TRAINING ---
        model.train()
        running_loss = 0.0
        total_graphs = 0  # Contatore dinamico per evitare errori di tipo con Pylance
        
        for batch_data in train_loader:
            batch_data = batch_data.to(torch.device(cfg.DEVICE))
            optimizer.zero_grad()  # Resetta i gradienti del passo precedente
            
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
    return evaluate_model(model, val_loader)

def train_final_model(
    dataset,
    model_class,
    lr: float,
    dropout_p: float,
    hidden_channels: int = cfg.DEFAULT_HIDDEN_CHANNELS,
    epochs: int = cfg.DEFAULT_EPOCHS,
    batch_size: int = cfg.DEFAULT_BATCH_SIZE,
    noise_level: float = cfg.NOISE_LEVEL,
    weight_decay: float = 1e-4,
    excluded_features=None
):
    """
    Addestra il modello definitivo sull'intero set di Training+Validation (850 grafi).
    Soddisfa il requisito di addestramento finale e tracciamento della loss per i grafici.

    Returns
    -------
    tuple
        model: Il modello addestrato definitivo.
        loss_history: Lista delle loss per ogni epoca (utile per il plot).
        mean: Media delle feature calcolata sull'intero dataset di train+val.
        std: Deviazione standard delle feature calcolata sull'intero dataset di train+val.
    """
    # Calcolo delle statistiche Z-Score sull'intero blocco di addestramento
    all_x = torch.cat([g.x for g in dataset], dim=0)
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0)
    std[std == 0] = 1.0

    # Applicazione della normalizzazione Z-Score 
    dataset_cp = dl.apply_z_score(dataset, mean, std, excluded_features)

    # Applicazione della Noise Injection DOPO la Z-Score per uniformare l'intensità del rumore
    if cfg.NOISE_LEVEL > 0.0:
        dataset_cp = dl.inject_node_noise(dataset_cp, noise_level=noise_level, excluded_features=excluded_features)

    # 3. Creazione del DataLoader finale (con shuffle e drop_last consistenti)
    final_loader = DataLoader(dataset_cp, batch_size=batch_size, shuffle=True, drop_last=False)

    # 4. Inizializzazione del modello con le feature dinamiche
    num_features = all_x.shape[1]
    model = model_class(in_channels=num_features, hidden_channels=hidden_channels, dropout_p=dropout_p)
    model.to(cfg.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []

    print(f"[INFO] Avvio Addestramento Finale del modello {model_class.__name__}...")
    print(f"       Configurazione: LR={lr} | Dropout={dropout_p} | Epoche={epochs} | Canali={hidden_channels} | Rumore={noise_level}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_graphs = 0

        for batch_data in final_loader:
            batch_data = batch_data.to(cfg.DEVICE)
            optimizer.zero_grad()
            
            logits = model(batch_data).squeeze(-1)
            labels = batch_data.y.float()
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_data.num_graphs
            total_graphs += batch_data.num_graphs

        epoch_loss = running_loss / total_graphs
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"   [Epoca {epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f}")

    print(f"\n[INFO] Addestramento finale completato con successo.")
    return model, loss_history, mean, std