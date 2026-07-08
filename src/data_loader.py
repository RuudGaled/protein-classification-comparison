import copy
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src import config as cfg
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from scipy.stats import skew

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

def inspect_node_features(dataset):
    """
    Analizza la natura delle feature dei nodi calcolando le statistiche descrittive.

    Restituisce la lista delle feature asimmetriche e un DataFrame riassuntivo.
    """

    all_x = torch.cat([graph.x for graph in dataset], dim=0)
    print("\n===== FEATURE INSPECTION AND SKEWNESS ANALYSIS =====")

    # Indici delle feature continue con forte asimmetria
    highly_skewed_features = []
    stats_records = []

    for col_idx in range(all_x.shape[1]):
        column = all_x[:, col_idx]

        unique_values = torch.unique(column)
        unique_count = unique_values.numel()

        feature_min = column.min().item()
        feature_max = column.max().item()
        mean = column.mean().item()
        std = column.std().item()
        median = column.median().item()

        # Calcolo della Skewness 
        feat_skew = skew(column.cpu().numpy())

        print(
            f"Feature {col_idx:02d} | "
            f"min={feature_min:.2f} | max={feature_max:.2f} | "
            f"mean={mean:.2f} | median={median:.2f} | "
            f"std={std:.2f} | skew={feat_skew:.2f} | unique={unique_count}"
        )

        # Salvataggio record per il DataFrame
        stats_records.append({
            "Feature": col_idx, "Min": feature_min, "Max": feature_max,
            "Mean": mean, "Median": median, "Std": std, 
            "Skewness": feat_skew, "Unique": unique_count
        })

        # Se la skewness è maggiore di 1.5 e non è una variabile binaria/one-hot
        if abs(feat_skew) > 1.5 and unique_count > 10:
            highly_skewed_features.append(col_idx)

    print("====================================================\n")

    df_stats = pd.DataFrame(stats_records)

    return highly_skewed_features, df_stats

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

def compute_safe_log_shifts(train_graphs, features_to_transform):
    """
    Calcola, per ciascuna feature selezionata, lo shift minimo necessario
    affinché tutti i valori osservati nel training risultino maggiori o
    uguali a 1 prima dell'applicazione della trasformazione logaritmica.

    Gli shift vengono calcolati esclusivamente sul training set per
    evitare data leakage.

    Parameters
    ----------
    train_graphs : list[torch_geometric.data.Data]
        Lista dei grafi del training set.
    features_to_transform : iterable[int]
        Indici delle feature continue da trasformare.

    Returns
    -------
    dict[int, float]
        Dizionario {feature_index: shift}.
    """
    all_train_x = torch.cat([graph.x for graph in train_graphs], dim=0)

    shifts = {}

    for idx in features_to_transform:
        min_value = all_train_x[:, idx].min().item()

        if min_value <= 0:
            # Shift di |min| + 1 per garantire che il valore minimo inserito nel log sia >= 1
            shifts[idx] = -min_value + 1.0
        else:
            shifts[idx] = 0.0

    return shifts


def apply_safe_log_transform(graphs, features_to_transform, shifts):
    """
    Restituisce una copia dei grafi applicando la trasformazione

        log(x + shift)

    alle feature selezionate, utilizzando gli shift calcolati sul training
    set.

    Parameters
    ----------
    graphs : list[torch_geometric.data.Data]
        Grafi da trasformare.
    features_to_transform : iterable[int]
        Indici delle feature da trasformare.
    shifts : dict[int, float]
        Dizionario contenente gli shift calcolati sul training set.

    Returns
    -------
    list[torch_geometric.data.Data]
        Nuova lista di grafi trasformati.

    Raises
    ------
    KeyError
        Se manca lo shift per una feature richiesta.

    ValueError
        Se qualche valore risulta non positivo dopo l'applicazione dello
        shift, rendendo la trasformazione logaritmica non definita.
    """
    transformed_graphs = [graph.clone() for graph in graphs]

    for graph in transformed_graphs:
        for idx in features_to_transform:

            if idx not in shifts:
                raise KeyError(f"Missing log shift for feature {idx}.")

            values = graph.x[:, idx] + shifts[idx]

            # Invece di sollevare errore, clippiamo i valori anomali del validation
            # a un numero piccolissimo positivo per permettere il logaritmo.
            values = torch.clamp(values, min=1e-6)

            graph.x[:, idx] = torch.log(values)

    return transformed_graphs

def apply_z_score(graphs, mean, std, excluded_features=None):
    """
    Applica una normalizzazione Z-Score a una lista di grafi usando medie e std precalcolate,
    rispettando la maschera delle feature da escludere.
    """
    graphs_cp = [graph.clone() for graph in graphs]
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

def inject_node_noise(dataset, noise_level: float = cfg.NOISE_LEVEL, excluded_features=None):
    """
    Applica la Noise Injection (Data Augmentation) sulle feature dei nodi.
    Genera un rumore gaussiano a media zero per aumentare la robustezza del modello, escludendo le feature discrete tramite maschera.
    """
    # dataset_augmented = [copy.deepcopy(g) for g in dataset]
    dataset_augmented = [graph.clone() for graph in dataset]
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

def drop_features_permanently(dataset, indexes_to_drop):
    """
    Restituisce una copia del dataset in cui le feature indicate
    sono state rimosse definitivamente dalla matrice ``x`` di ogni grafo.
    """
    dataset_trimmed = []
    total_features = dataset[0].x.shape[1]

    # Converte gli indici in un set per velocizzare i test di appartenenza
    indexes_to_drop = set(indexes_to_drop)

    # Controllo validità indici feature da rimuovere
    if any(i < 0 or i >= total_features for i in indexes_to_drop):
        raise ValueError(
            "Gli indici delle feature da rimuovere non sono validi."
        )

    # Costruisce la lista delle feature da mantenere
    keep_indexes = [
        i for i in range(total_features)
        if i not in indexes_to_drop
    ]
    
    for graph in dataset:
        graph_copy = copy.deepcopy(graph)
        
        # Rimuove definitivamente le colonne corrispondenti alle feature escluse
        graph_copy.x = graph_copy.x[:, keep_indexes]
        dataset_trimmed.append(graph_copy)
        
    return dataset_trimmed