import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import dei moduli locali dall'infrastruttura src
from src import config as cfg
from src import utils as ut
from src import data_loader as dl
from src import train as tr
from src import models as mdl

def main():
    print("\n" + "="*71)
    print(" PIPELINE PROGETTO DEEP LEARNING - CLASSIFICAZIONE PROTEINE ")
    print("="*71)
    
    # Setup Riproducibilità e Device
    ut.setup_reproducibility(cfg.FIXED_SEED)
    device = ut.get_device()
    print(f"[INFO] Addestramento rete neurale attraverso: {device}")

    # Creazione dinamica della cartella dei risultati
    results_dir = cfg.PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Caricamento e validazione dataset
    dataset = dl.load_and_validate_dataset(str(cfg.DATASET_PATH))

    # Analisi features
    skewed_features, df_feature_stats = dl.inspect_node_features(dataset)

    # Generazione ed esportazione del grafico delle distribuzioni
    ut.plot_skewed_features_distributions(
        dataset=dataset, 
        feature_indices=skewed_features, 
        save_path=cfg.RESULTS_DIR / "node_features_distributions.png",
        show = False
    )

    # Analisi dello sbilanciamento delle classi
    dl.check_class_imbalance(dataset)

    # Split dataset (Isolamento sicuro del Test Set)
    train_val_dataset, test_dataset = dl.stratified_holdout_split(dataset, cfg.TEST_SIZE, cfg.FIXED_SEED)

    # Griglia di iperparametri
    grid_lr = [1e-3, 1e-4]
    grid_dropout = [0.1, 0.3]

    # Lista di indici delle feature da non normalizzare
    excluded_indexes = [4, 29, 30, 31]

    results_archive = {}

    # Model selection tramite Cross-Validation e Grid Search
    print("\n" + "="*71)
    print(" AVVIO GRID SEARCH: GCN & GAT")
    print("="*71)

    print("\n[INFO]: Avvio procedura di Cross Validation...")
    print("[INFO]: Impostata applicazione Z-Score e Noise Injection\n    in modo isolato all'interno di ciascun fold.\n")
    
    for model_name, model_class in [("GCN", mdl.ProteinGCN), ("GAT", mdl.ProteinGAT)]:
        print(f"\n VALUTAZIONE ARCHITETTURA: Protein{model_name}")
        for lr in grid_lr:
            for drop in grid_dropout:
                config_name = f"{model_name}_lr_{lr}_drop_{drop}"
                print(f"\n>> Testando configurazione: {config_name}")
                
                f1_res, auc_res = tr.run_cross_validation(
                    dataset=train_val_dataset,
                    model_class=model_class,
                    n_splits=cfg.N_SPLITS,
                    batch_size=cfg.DEFAULT_BATCH_SIZE,
                    train_fold_fn=tr.train_single_fold,
                    seed=cfg.FIXED_SEED,
                    lr=lr,
                    dropout_p=drop,
                    hidden_channels=cfg.DEFAULT_HIDDEN_CHANNELS,
                    excluded_features=excluded_indexes,
                    noise_level=cfg.NOISE_LEVEL,
                    use_log_transform=False,
                )
                results_archive[config_name] = {
                    "model": model_name, "lr": lr, "dropout": drop, "channels": cfg.DEFAULT_HIDDEN_CHANNELS,
                    "f1_mean": np.mean(f1_res), "f1_std": np.std(f1_res),
                    "auc_mean": np.mean(auc_res), "auc_std": np.std(auc_res),
                    "f1_scores": f1_res, "auc_scores": auc_res,
                }

    # print("\n[INFO]: Lavoro completato.")

    # Compilazione ed esportazione della Tabella Comparativa
    df_results = pd.DataFrame.from_dict(results_archive, orient="index")
    df_results = df_results.sort_values(by="f1_mean", ascending=False)
    
    # Esportazione della tabella in CSV per massima trasparenza accademica
    df_results.to_csv(results_dir / "grid_search_results.csv", float_format="%.4f")
    
    # Stampa su terminale
    print("\n" + "-"*20 + " GRADUATORIA CONFIGURAZIONI " + "-"*23)
    display_results = df_results.drop(columns=["model", "lr", "dropout", "channels", "f1_scores", "auc_scores"])
    with pd.option_context('display.max_columns', None, 'display.width', 1000):
        print(display_results.round(4).to_string(col_space=12))
    print("-"*71)

    # Estrazione automatica del modello vincente
    selected_config = df_results.iloc[0]
    best_model_class = mdl.ProteinGCN if selected_config["model"] == "GCN" else mdl.ProteinGAT
    
    print(f"\n[CONFIGURAZIONE VINCITRICE SELEZIONATA]")
    print(f" -> Stringa identificativa: {selected_config.name}")
    print(f" -> Architettura:           Protein{selected_config['model']}")
    print(f" -> Learning Rate:          {selected_config['lr']}")
    print(f" -> Dropout:                {selected_config['dropout']}")
    print(f" -> larghezza canali :      {selected_config['channels']}")
    print(f" -> Macro F1 medio in CV:   {selected_config['f1_mean']:.4f} ± {selected_config['f1_std']:.4f}")
    print(f" -> ROC-AUC medio in CV :   {selected_config['auc_mean']:.4f} ± {selected_config['auc_std']:.4f}")

    # TEST 1: Log Transform
    print("\n=======================================================================")
    print("Sperimentazione Log Trasform + Z-Score Selettiva")
    print("=======================================================================")
    print("[DESCRIZIONE]: Verifica se la trasformazione logaritmica delle feature continue\n        migliora l'addestramento mitigando l'asimmetria (skewness).\n")

    f1_log, auc_log = tr.run_cross_validation(
        dataset=train_val_dataset, model_class=best_model_class, n_splits=cfg.N_SPLITS,
        batch_size=cfg.DEFAULT_BATCH_SIZE, train_fold_fn=tr.train_single_fold, seed=cfg.FIXED_SEED,
        lr=selected_config['lr'], dropout_p=selected_config['dropout'],
        hidden_channels=cfg.DEFAULT_HIDDEN_CHANNELS, excluded_features=excluded_indexes,
        noise_level=cfg.NOISE_LEVEL, use_log_transform=True
    )

    print(f"\n -> Performance CON Log-Transform: Macro F1 Media={np.mean(f1_log):.4f} | ROC-AUC Media={np.mean(auc_log):.4f}")
    print(f" -> Performance SENZA Log-Transform: Macro F1 Media={selected_config['f1_mean']:.4f} | ROC-AUC Media={selected_config['auc_mean']:.4f}")

    delta_f1 = np.mean(f1_log) - selected_config["f1_mean"]
    delta_auc = np.mean(auc_log) - selected_config["auc_mean"]

    print(
        f"\nΔ Macro F1 Medio: {delta_f1:+.4f} ({delta_f1 / selected_config['f1_mean'] * 100:+.2f}%)"
    )

    print(
        f"Δ ROC-AUC Medio: {delta_auc:+.4f} ({delta_auc / selected_config['auc_mean'] * 100:+.2f}%)"
    )

    ut.plot_cv_comparison(
        baseline_scores=np.array([selected_config["f1_mean"], selected_config["auc_mean"]]),
        comparison_scores=np.array([np.mean(f1_log), np.mean(auc_log)]),
        baseline_std=np.array([selected_config["f1_std"], selected_config["auc_std"]]),
        comparison_std=np.array([np.std(f1_log), np.std(auc_log)]),
        baseline_fold_scores=[selected_config["f1_scores"], selected_config["auc_scores"]],
        comparison_fold_scores=[f1_log, auc_log], 
        comparison_label="Log Transform",
        title="Baseline vs Log Transform", 
        save_path=cfg.RESULTS_DIR / "test_log_transform.png", 
        show=False
    )

    print("\n[INFO] Esportazione grafico relativo alla trasformazione logaritmica completata.\n")

    if np.mean(f1_log) > selected_config['f1_mean']:
        final_use_log = True

        current_baseline_f1 = np.mean(f1_log)
        current_baseline_auc = np.mean(auc_log)
        current_baseline_f1_std = np.std(f1_log)
        current_baseline_auc_std = np.std(auc_log)
        current_baseline_f1_scores = f1_log
        current_baseline_auc_scores = auc_log
        current_baseline_label = "Log Transform (Nuova Baseline)"

        print("[+] Log-Transform: ATTIVATA (Ha migliorato le performance)")
    else:
        final_use_log = False

        current_baseline_f1 = selected_config['f1_mean']
        current_baseline_auc = selected_config['auc_mean']
        current_baseline_f1_std = selected_config['f1_std']
        current_baseline_auc_std = selected_config['auc_std']
        current_baseline_f1_scores = selected_config['f1_scores']
        current_baseline_auc_scores = selected_config['auc_scores']
        current_baseline_label = "Grid Search Baseline"
        
        print("[-] Log-Transform: DISATTIVATA (Nessun beneficio)")

    # TEST 2: Rimozione delle feature one-hot
    print("\n=======================================================================")
    print("Test di Esclusione delle Feature One-Hot")
    print("=======================================================================")
    print("[DESCRIZIONE]: Valuta se le 3 feature della struttura secondaria aggiungono segnale\n        utile o se risultano ridondanti per la classificazione.\n")

    dataset_without_onehot = dl.drop_features_permanently(train_val_dataset, indexes_to_drop=[29, 30, 31])

    f1_no_onehot, auc_no_onehot = tr.run_cross_validation(
        dataset=dataset_without_onehot, 
        model_class=best_model_class, 
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.DEFAULT_BATCH_SIZE, 
        train_fold_fn=tr.train_single_fold, 
        seed=cfg.FIXED_SEED,
        lr=selected_config['lr'], 
        dropout_p=selected_config['dropout'],
        hidden_channels=cfg.DEFAULT_HIDDEN_CHANNELS, 
        excluded_features=[4],
        noise_level=cfg.NOISE_LEVEL, 
        use_log_transform=final_use_log
    )

    print(f"\n -> Modello Base Attuale (32 features): Macro F1 Media={current_baseline_f1:.4f} | ROC-AUC Media={current_baseline_auc:.4f}")
    print(f" -> Modello No One-Hot (29 features):   Macro F1 Media = {np.mean(f1_no_onehot):.4f} | ROC-AUC Media={np.mean(auc_no_onehot):.4f}")

    delta_f1 = np.mean(f1_no_onehot) - current_baseline_f1
    delta_auc = np.mean(auc_no_onehot) - current_baseline_auc

    print(f"\nΔ Macro F1 Medio: {delta_f1:+.4f} ({delta_f1 / current_baseline_f1 * 100:+.2f}%)")
    print(f"Δ ROC-AUC Medio: {delta_auc:+.4f} ({delta_auc / current_baseline_auc * 100:+.2f}%)")

    ut.plot_cv_comparison(
        baseline_scores=np.array([current_baseline_f1, current_baseline_auc]),
        comparison_scores=np.array([np.mean(f1_no_onehot), np.mean(auc_no_onehot)]),
        baseline_std=np.array([current_baseline_f1_std, current_baseline_auc_std]),
        comparison_std=np.array([np.std(f1_no_onehot), np.std(auc_no_onehot)]),
        baseline_fold_scores=[current_baseline_f1_scores, current_baseline_auc_scores],
        comparison_fold_scores=[f1_no_onehot, auc_no_onehot], comparison_label="No One-Hot",
        title=f"{current_baseline_label} vs No One-Hot Feature",
        save_path=cfg.RESULTS_DIR / "test_no_onehot.png",
        show=False
    )

    print("\n[INFO] Esportazione grafico relativo alla rimozione feature one-hot completata.\n")

    if np.mean(f1_no_onehot) > current_baseline_f1:
        # Utilizzo dataset "tagliato" e feature 4 unica da escludere dalla Z-Score
        final_dataset = dataset_without_onehot 
        final_excluded_indexes = [4] 

        current_baseline_f1 = np.mean(f1_no_onehot)
        current_baseline_auc = np.mean(auc_no_onehot)
        current_baseline_f1_std = np.std(f1_no_onehot)
        current_baseline_auc_std = np.std(auc_no_onehot)
        current_baseline_f1_scores = f1_no_onehot
        current_baseline_auc_scores = auc_no_onehot
        current_baseline_label = "No One-Hot (Nuova Baseline)"

        print("[+] Esclusione One-Hot: APPLICATA (Le feature erano ridondanti)")
    else:
        final_dataset = train_val_dataset
        final_excluded_indexes = [4, 29, 30, 31]
        print("[-] Esclusione One-Hot: NON APPLICATA (Le feature aggiungono segnale utile)")

    # TEST 3: Sensibilità Architetturale
    print("\n=======================================================================")
    print(f"Avvio analisi di sensibilità architetturale (hidden_channel: {cfg.ALT_HIDDEN_CHANNELS})")
    print("=======================================================================")
    print(f"[DESCRIZIONE]: Confronta le prestazioni variando la dimensionalità dei layer nascosti\n         per ottimizzare la complessità strutturale della rete.\n")

    f1_channels, auc_channels = tr.run_cross_validation(
        dataset=final_dataset, 
        model_class=best_model_class, 
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.DEFAULT_BATCH_SIZE, 
        train_fold_fn=tr.train_single_fold, 
        seed=cfg.FIXED_SEED,
        lr=selected_config['lr'], 
        dropout_p=selected_config['dropout'],
        hidden_channels=cfg.ALT_HIDDEN_CHANNELS, excluded_features=final_excluded_indexes,
        noise_level=cfg.NOISE_LEVEL, 
        use_log_transform=final_use_log
    )

    print(f"\n -> Performance Modello Base Attuale: Macro F1 Media={current_baseline_f1:.4f} | ROC-AUC Media={current_baseline_auc:.4f}")
    print(f" -> Performance con Channels={cfg.ALT_HIDDEN_CHANNELS}:   Macro F1 Media = {np.mean(f1_channels):.4f} | ROC-AUC Media={np.mean(auc_channels):.4f}")

    delta_f1 = np.mean(f1_channels) - current_baseline_f1
    delta_auc = np.mean(auc_channels) - current_baseline_auc

    print(f"\nΔ Macro F1 Medio: {delta_f1:+.4f} ({delta_f1 / current_baseline_f1 * 100:+.2f}%)")
    print(f"Δ ROC-AUC Medio: {delta_auc:+.4f} ({delta_auc / current_baseline_auc * 100:+.2f}%)")

    ut.plot_cv_comparison(
        baseline_scores=np.array([current_baseline_f1, current_baseline_auc]),
        comparison_scores=np.array([np.mean(f1_channels), np.mean(auc_channels)]),
        baseline_std=np.array([current_baseline_f1_std, current_baseline_auc_std]),
        comparison_std=np.array([np.std(f1_channels), np.std(auc_channels)]),
        baseline_fold_scores=[current_baseline_f1_scores, current_baseline_auc_scores],
        comparison_fold_scores=[f1_channels, auc_channels], comparison_label=f"Channels={cfg.ALT_HIDDEN_CHANNELS}",
        title=f"{current_baseline_label} vs Channels={cfg.ALT_HIDDEN_CHANNELS}", 
        save_path=cfg.RESULTS_DIR / "test_alt_hidden_channels.png", 
        show=False
    )

    print("\n[INFO] Esportazione grafico sensibilità architetturale completata.\n")

    if np.mean(f1_channels) > current_baseline_f1:
        final_channels = cfg.ALT_HIDDEN_CHANNELS
        print(f"[+] STRUTTURA AGGIORNATA: Scelti {cfg.ALT_HIDDEN_CHANNELS} canali per l'addestramento finale.")
    else:
        final_channels = cfg.DEFAULT_HIDDEN_CHANNELS
        print(f"[-] STRUTTURA CONFERMATA: Mantenuti i {cfg.DEFAULT_HIDDEN_CHANNELS} canali di default.")

    # Addestramento definitivo con configurazione migliore
    print("\n" + "="*71)
    print("AVVIO ADDESTRAMENTO FINALE")
    print("="*71)
    print("\n[INFO]: Configurazione architetturale e pipeline di preprocessing definitiva:")

    log_status = "Attiva" if final_use_log else "Disattiva"
    onehot_status = "Applicata (29 feat)" if len(final_excluded_indexes) == 1 else "Non Applicata (32 feat)"

    print("-" * 55)
    print(f" {'PARAMETRO':<20} | {'VALORE'}")
    print("-" * 55)
    print(f" {'Modello':<20} | {selected_config['model']}")
    print(f" {'Learning Rate':<20} | {selected_config['lr']}")
    print(f" {'Dropout':<20} | {selected_config['dropout']}")
    print(f" {'Hidden Channels':<20} | {final_channels}")
    print(f" {'Epoche':<20} | {cfg.DEFAULT_EPOCHS}")
    print(f" {'Noise Level':<20} | {cfg.NOISE_LEVEL}")
    print(f" {'Log-Transform':<20} | {log_status}")
    print(f" {'Rimozione One-Hot':<20} | {onehot_status}")
    print("-" * 55 + "\n")

    # Addestramento finale del modello 
    final_model, history_loss, shifts, final_mean, final_std = tr.train_final_model(    
        dataset=final_dataset, 
        model_class=best_model_class, 
        lr=selected_config['lr'],
        dropout_p=selected_config['dropout'], 
        hidden_channels=final_channels, 
        epochs=cfg.DEFAULT_EPOCHS,
        batch_size=cfg.DEFAULT_BATCH_SIZE, 
        noise_level=cfg.NOISE_LEVEL, 
        weight_decay=1e-4,
        excluded_features=final_excluded_indexes, use_log_transform=final_use_log
    )

    # Valutazione sul Test Set
    if len(final_excluded_indexes) == 1: 
        test_dataset = dl.drop_features_permanently(test_dataset, indexes_to_drop=[29, 30, 31])

    if shifts is not None:
        continuous_features = [i for i in range(test_dataset[0].x.shape[1]) if i not in final_excluded_indexes]
        test_dataset = dl.apply_safe_log_transform(test_dataset, continuous_features, shifts)

    test_dataset_normalized = dl.apply_z_score(test_dataset, final_mean, final_std, excluded_features=final_excluded_indexes)
    test_loader = DataLoader(test_dataset_normalized, batch_size=cfg.DEFAULT_BATCH_SIZE, shuffle=False, drop_last=False)

    y_test_true, y_test_pred, y_test_score = tr.evaluate_model(final_model, test_loader)

    acc = accuracy_score(y_test_true, y_test_pred)
    f1_macro = f1_score(y_test_true, y_test_pred, average="macro")
    precision = precision_score(y_test_true, y_test_pred, zero_division=0)
    recall = recall_score(y_test_true, y_test_pred)
    roc_auc = roc_auc_score(y_test_true, y_test_score)

    final_metrics = pd.DataFrame({"Accuracy": [acc], "Macro_F1": [f1_macro], "Precision": [precision], "Recall": [recall], "ROC_AUC": [roc_auc]})
    final_metrics.to_csv(cfg.RESULTS_DIR / "final_test_metrics.csv", index=False, float_format="%.4f")

    print("\n=============================================================")
    print(" METRICHE DEFINITIVE SUL TEST SET (MODELLO VINCENTE) ")
    print("=============================================================")
    print(f" - Accuracy  : {acc:.4f}")
    print(f" - Macro F1  : {f1_macro:.4f}")
    print(f" - Precision : {precision:.4f}")
    print(f" - Recall    : {recall:.4f}")
    print(f" - ROC-AUC   : {roc_auc:.4f}")
    print("=============================================================\n")

    ut.plot_final_evaluation_metrics(
        history_loss=history_loss, 
        y_true=y_test_true, 
        y_pred=y_test_pred, 
        y_score=y_test_score,
        roc_auc=float(roc_auc), 
        save_path=cfg.RESULTS_DIR / "final_evaluation_plots.png", 
        show=False
    )

    print(f"\n[OK] Pipeline completata. Artefatti salvati in: {cfg.RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    main()