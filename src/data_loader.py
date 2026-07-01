import copy
import os
import torch
from src import config as cfg
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

def load_and_validate_dataset(file_path: str):
    """
    Carica il dataset e verifica l'assenza di valori mancanti (NaN).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset non trovato al percorso: {file_path}")
        
    dataset = torch.load(file_path, weights_only=False)
    
    # Controllo formale anti-NaN per i nodi
    has_nan = any(torch.isnan(data.x).any() for data in dataset)
    assert not has_nan, "[ERRORE] Il dataset contiene valori NaN nelle feature dei nodi!"
    
    print(f"[INFO] Dataset caricato con successo. Numero totale di grafi: {len(dataset)}")
    return dataset

def check_class_imbalance(dataset):
    """
    Analizza e stampa la distribuzione delle classi (Target y) nel dataset.
    """
    labels = [data.y.item() for data in dataset]
    total = len(labels)
    
    # Conta le classi (0: non-enzima, 1: enzima)
    class_0 = labels.count(0)
    class_1 = labels.count(1)
    
    print("\n--- ANALISI BILANCIAMENTO DATASET ---")
    print(f"Classe 0 (Non-Enzimi): {class_0} ({(class_0/total)*100:.2f}%)")
    print(f"Classe 1 (Enzimi)    : {class_1} ({(class_1/total)*100:.2f}%)")
    print("-------------------------------------\n")
    
    # Restituisce un dizionario con il conteggio
    return {0: class_0, 1: class_1}

def stratified_holdout_split(dataset, 
                             test_size: float = cfg.TEST_SIZE, 
                             seed: int = cfg.FIXED_SEED):
    """
    Isola un Test Set definitivo usando uno split stratificato per evitare Data Leakage.
    """
    labels = [data.y.item() for data in dataset]
    
    train_val_set, test_set = train_test_split(
        dataset, 
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )
    print(f"[INFO] Split Dataset completato: Train+Val = {len(train_val_set)} grafi, Test = {len(test_set)} grafi.")
    return train_val_set, test_set

def apply_z_score(graphs, mean, std, excluded_features=None):
    """
    Applica una normalizzazione Z-Score a una lista di grafi usando medie e std precalcolate,
    rispettando la maschera delle feature da escludere.
    """
    graphs_cp = [copy.deepcopy(g) for g in graphs]
    n_features = graphs_cp[0].x.shape[1]
    
    if excluded_features is None:
        features_to_normalize = list(range(n_features))
    else:
        features_to_normalize = [i for i in range(n_features) if i not in excluded_features]
        
    for g in graphs_cp:
        g.x[:, features_to_normalize] = (
            g.x[:, features_to_normalize] - mean[features_to_normalize]
        ) / std[features_to_normalize]
        
    return graphs_cp

def normalize_fold_data(train_graphs, val_graphs, excluded_features=None):
    """
    Applica la normalizzazione Z-Score calcolando le statistiche sul train
    e riutilizzandole sul validation tramite apply_z_score.
    """
    all_train_x = torch.cat([g.x for g in train_graphs], dim=0)
    mean = all_train_x.mean(dim=0)
    std = all_train_x.std(dim=0)
    std[std == 0] = 1.0
    
    train_graphs_cp = apply_z_score(train_graphs, mean, std, excluded_features)
    val_graphs_cp = apply_z_score(val_graphs, mean, std, excluded_features)
    
    return train_graphs_cp, val_graphs_cp

def create_dataloaders(train_set, val_set, batch_size: int = 32):
    """
    Incapsula le liste di grafi negli oggetti DataLoader nativi di PyTorch Geometric.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def inspect_node_features(dataset):
    """
    Analizza la natura delle feature dei nodi.

    Restituisce statistiche utili per decidere
    quali feature normalizzare.
    """

    all_x = torch.cat(
        [graph.x for graph in dataset],
        dim=0
    )

    print("\n===== FEATURE INSPECTION =====")

    for col_idx in range(all_x.shape[1]):

        column = all_x[:, col_idx]

        unique_values = torch.unique(column)

        feature_min = column.min().item()
        feature_max = column.max().item()

        mean = column.mean().item()
        std = column.std().item()

        median = column.median().item()

        print(
            f"Feature {col_idx:02d} | "
            f"min={feature_min:.4f} | "
            f"max={feature_max:.4f} | "
            f"mean={mean:.4f} | "
            f"std={std:.4f} | "
            f"median={median:.4f} | "
            f"unique={len(unique_values)}"
        )

    # print("\nAnalisi specifica sui tipi dei valori presenti nelle feature 09-20\n")

    # for i in range(9, 21):
    #     col = all_x[:, i]

    #     print(
    #         i,
    #         torch.allclose(col, col.round())
    #     )   

    print("==============================\n")

def inject_node_noise(dataset, noise_level: float = 0.01, excluded_features=None):
    """
    Applica la Noise Injection (Data Augmentation) sulle feature dei nodi.
    Genera un rumore gaussiano a media zero per aumentare la robustezza del modello, escludendo le feature discrete tramite maschera.
    """
    dataset_augmented = [copy.deepcopy(g) for g in dataset]
    n_features = dataset_augmented[0].x.shape[1]

    if excluded_features is None:
        features_to_noise = list(range(n_features))
    else:
        features_to_noise = [i for i in range(n_features) if i not in excluded_features]
    
    for g in dataset_augmented:
        # Si genera e inietta il rumore solo sulle colonne destinate
        noise = torch.randn_like(g.x[:, features_to_noise]) * noise_level
        g.x[:, features_to_noise] += noise
        
    return dataset_augmented