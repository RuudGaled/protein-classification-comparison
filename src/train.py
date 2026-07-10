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
    use_log_transform=False,
):
    """
    Esegue una Stratified K-Fold Cross Validation su un dataset di grafi.

    Per ciascun fold vengono eseguite, nell'ordine:

    1. eventuale trasformazione logaritmica delle feature continue,
       stimando gli shift esclusivamente sul training fold;
    2. normalizzazione Z-Score utilizzando media e deviazione standard
       calcolate sul training fold;
    3. applicazione Noise Injection sul training fold;
    4. addestramento del modello;
    5. valutazione sul validation fold.
    """
    if train_fold_fn is None:
        raise ValueError(
            "[ERRORE] È necessario fornire una funzione di training "
            "(train_fold_fn)."
        )

    # Dimensione delle feature dei nodi 
    # (tutti i grafi hanno lo stesso numero di feature per nodo)
    n_features = dataset[0].x.shape[1]

    excluded = set(excluded_features or [])

    # Controllo validità indici feature da escludere
    invalid_features = [
        idx for idx in excluded
        if idx < 0 or idx >= n_features
    ]

    if invalid_features:
        raise ValueError(
            f"Indici di feature esclusi non validi: "
            f"{sorted(invalid_features)}."
        )

    # Feature continue sulle quali applicare le trasformazioni
    continuous_features = [
        idx for idx in range(n_features)
        if idx not in excluded
    ]

    labels = [graph.y.item() for graph in dataset]
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    f1_scores = []
    auc_scores = []
    dummy_X = np.zeros(len(labels))

    for fold_id, (train_idx, val_idx) in enumerate(
        skf.split(dummy_X, labels),
        start=1,
    ):
        print(f"\n   --- FOLD {fold_id}/{n_splits} ---")

        # Split dei grafi per il fold corrente
        train_graphs = [dataset[i] for i in train_idx]
        val_graphs = [dataset[i] for i in val_idx]

        # Applicazione trasformazione logaritmica
        if use_log_transform:
            shifts = dl.compute_safe_log_shifts(
                train_graphs,
                continuous_features,
            )
            train_graphs = dl.apply_safe_log_transform(
                train_graphs,
                continuous_features,
                shifts,
            )
            val_graphs = dl.apply_safe_log_transform(
                val_graphs,
                continuous_features,
                shifts,
            )
            
        # Normalizzazione Z-Score
        train_graphs, val_graphs = dl.normalize_fold_data(
            train_graphs,
            val_graphs,
            excluded_features=excluded,
        )

        # Applicazione Noise Injection
        train_graphs = dl.inject_node_noise(
            train_graphs,
            noise_level=noise_level,
            excluded_features=excluded,
        )

        # Creazione dei DataLoader nativi PyG
        train_loader, val_loader = dl.create_dataloaders(
            train_graphs,
            val_graphs,
            batch_size=batch_size,
        )

        # Creazione istanza modello
        model = model_class(
            in_channels=n_features,
            hidden_channels=hidden_channels,
            dropout_p=dropout_p,
        )

        # Training sul singolo fold
        y_true, y_pred, y_score = train_fold_fn(
            model,
            train_loader,
            val_loader,
            lr=lr,
            epochs=cfg.DEFAULT_EPOCHS,
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
    """
    # Funzione di loss e ottimizzatore (regolarizzazione L2 tramite Weight Decay)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Spostia esplicitamente il modello su CPU 
    model.to(cfg.DEVICE)

    # Ciclo di addestramento
    for epoch in range(1, epochs + 1):
        # Fase di training
        model.train()
        running_loss = 0.0
        #  Numero complessivo di grafi elaborati nell'epoca
        total_graphs = 0  
        
        for batch_data in train_loader:
            batch_data = batch_data.to(torch.device(cfg.DEVICE))
            # Resetta i gradienti del passo precedente
            optimizer.zero_grad()  
            
            # Logit prodotti dal modello
            logits = model(batch_data).squeeze(-1)
            # Target reale convertito in float per la loss 
            labels = batch_data.y.float()         
            
            # Calcolo dell'errore
            loss = criterion(logits, labels)    
            loss.backward()
            # Aggiornamento dei pesi
            optimizer.step()              
            
            running_loss += loss.item() * batch_data.num_graphs
            total_graphs += batch_data.num_graphs

        epoch_loss = running_loss / total_graphs
        
        # Stampa periodica della loss per monitorare l'addestramento
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"   [Epoca {epoch:02d}/{epochs:02d}] Training Loss: {epoch_loss:.4f}")

    # Valutazione finale del modello sul validation fold
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
    excluded_features=None,
    use_log_transform=False,
):
    """
    Addestra il modello definitivo sull'intero dataset di Training+Validation e restituisce i parametri di preprocessing necessari per applicare la stessa pipeline al test set.

    La pipeline di preprocessing è identica a quella utilizzata durante la
    Cross Validation:

    1. eventuale trasformazione logaritmica delle feature continue,
       stimando gli shift sull'intero dataset di Training+Validation;
    2. normalizzazione Z-Score;
    3. Noise Injection;
    4. addestramento del modello.
    """
    # Numero di feature per nodo
    num_features = dataset[0].x.shape[1]

    excluded = set(excluded_features or [])

    # Controllo validità indici feature da escludere
    invalid_features = [
        idx for idx in excluded
        if idx < 0 or idx >= num_features
    ]

    if invalid_features:
        raise ValueError(
            f"Indici di feature esclusi non validi: "
            f"{sorted(invalid_features)}."
        )

    # Feature continue sulle quali applicare le trasformazioni
    continuous_features = [
        idx for idx in range(num_features)
        if idx not in excluded
    ]

    train_graphs = dataset
    shifts = None

    # Applicazione trasformazione logaritmica
    if use_log_transform:
        shifts = dl.compute_safe_log_shifts(
            train_graphs,
            continuous_features,
        )
        train_graphs = dl.apply_safe_log_transform(
            train_graphs,
            continuous_features,
            shifts,
        )

    # Calcolo statistiche Z-Score sul dataset preprocessato
    all_x = torch.cat([graph.x for graph in train_graphs], dim=0)

    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0)
    std[std == 0] = 1.0

    # Normalizzazione Z-Score
    processed_dataset = dl.apply_z_score(
        train_graphs,
        mean,
        std,
        excluded,
    )

    # Applicazione Noise Injection
    processed_dataset = dl.inject_node_noise(
        processed_dataset,
        noise_level=noise_level,
        excluded_features=excluded,
    )

    # Creazione DataLoader finale
    final_loader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Creazione modello
    model = model_class(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        dropout_p=dropout_p,
    )

    model.to(cfg.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_history = []

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
            print(
                f"   [Epoca {epoch:02d}/{epochs:02d}] "
                f"Loss: {epoch_loss:.4f}"
            )

    print("\n[INFO] Addestramento finale completato con successo.")

    return model, loss_history, shifts, mean, std