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
    print(" PIPELINE INGEGNERIZZATA DI DEEP LEARNING - CLASSIFICAZIONE PROTEINE ")
    print("="*71)
    
    # 1. SETUP RIPRODUCIBILITÀ E DEVICE
    ut.setup_reproducibility(cfg.FIXED_SEED)
    device = ut.get_device()
    print(f"[INFO] Addestramento rete neurale attraverso: {device}")

    # Creazione dinamica della cartella dei risultati
    results_dir = cfg.PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Caricamento e validazione dataset
    dataset = dl.load_and_validate_dataset(str(cfg.DATASET_PATH))

    # Analisi features
    dl.inspect_node_features(dataset)

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
    
    # # Screening ProteinGCN
    # print("\n[INFO] Screening architettura: ProteinGCN")
    # for lr in grid_lr:
    #     for drop in grid_dropout:
    #         config_name = f"GCN_lr_{lr}_drop_{drop}"
    #         f1_res, auc_res = tr.run_cross_validation(
    #             dataset=train_val_dataset, 
    #             model_class=mdl.ProteinGCN,
    #             n_splits=cfg.N_SPLITS, 
    #             train_fold_fn=tr.train_single_fold,
    #             lr=lr, 
    #             dropout_p=drop, 
    #             excluded_features=excluded_indexes
    #         )
    #         results_archive[config_name] = {
    #             "model": "GCN", 
    #             "lr": lr, 
    #             "dropout": drop,
    #             "f1_mean": np.mean(f1_res), 
    #             "f1_std": np.std(f1_res),
    #             "auc_mean": np.mean(auc_res), 
    #             "auc_std": np.std(auc_res)
    #         }

    # # Screening ProteinGAT
    # print("\n[INFO] Screening architettura: ProteinGAT")
    # for lr in grid_lr:
    #     for drop in grid_dropout:
    #         config_name = f"GAT_lr_{lr}_drop_{drop}"
    #         f1_res, auc_res = tr.run_cross_validation(
    #             dataset=train_val_dataset, 
    #             model_class=mdl.ProteinGAT,
    #             n_splits=cfg.N_SPLITS, 
    #             train_fold_fn=tr.train_single_fold,
    #             lr=lr, 
    #             dropout_p=drop, 
    #             excluded_features=excluded_indexes
    #         )
    #         results_archive[config_name] = {
    #             "model": "GAT", 
    #             "lr": lr, 
    #             "dropout": drop,
    #             "f1_mean": np.mean(f1_res), 
    #             "f1_std": np.std(f1_res),
    #             "auc_mean": np.mean(auc_res), 
    #             "auc_std": np.std(auc_res)
    #         }
    
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
                    train_fold_fn=tr.train_single_fold,
                    lr=lr,
                    dropout_p=drop,
                    hidden_channels=cfg.DEFAULT_HIDDEN_CHANNELS,
                    excluded_features=excluded_indexes,
                    noise_level=cfg.NOISE_LEVEL
                )
                results_archive[config_name] = {
                    "model": model_name, "lr": lr, "dropout": drop, "channels": cfg.DEFAULT_HIDDEN_CHANNELS,
                    "f1_mean": np.mean(f1_res), "f1_std": np.std(f1_res),
                    "auc_mean": np.mean(auc_res), "auc_std": np.std(auc_res)
                }

    # print("\n[INFO]: Lavoro completato.")

    # Compilazione ed esportazione della Tabella Comparativa
    df_results = pd.DataFrame.from_dict(results_archive, orient="index")
    df_results = df_results.sort_values(by="f1_mean", ascending=False)
    
    # Esportazione della tabella in CSV per massima trasparenza accademica
    df_results.to_csv(results_dir / "grid_search_results.csv", float_format="%.4f")
    
    # Stampa su terminale
    print("\n" + "-"*20 + " GRADUATORIA CONFIGURAZIONI " + "-"*23)
    # print(df_results[["model", "lr", "dropout", "f1_mean", "auc_mean"]].round(4).to_string())
    # print(df_results.round(4).to_string())
    display_results = df_results.drop(columns=["model", "lr", "dropout", "channels"])
    with pd.option_context('display.max_columns', None, 'display.width', 1000):
        print(display_results.round(4).to_string(col_space=12))
    print("-"*71)

    # Estrazione automatica del modello vincente
    # best_config = df_results.index[0]
    # best_row = df_results.iloc[0]
    # best_lr = best_row["lr"]
    # best_drop = best_row["dropout"]
    # best_model_name = best_row["model"]
    # best_model_class = mdl.ProteinGCN if best_model_name == "GCN" else mdl.ProteinGAT
    selected_config = df_results.iloc[0]
    best_model_class = mdl.ProteinGCN if selected_config["model"] == "GCN" else mdl.ProteinGAT
    
    print(f"\n[CONFIGURAZIONE VINCITRICE SELEZIONATA]")
    print(f" -> Stringa identificativa: {selected_config.name}")
    print(f" -> Architettura:           Protein{selected_config['model']}")
    print(f" -> Learning Rate:          {selected_config['lr']}")
    print(f" -> Dropout:                {selected_config['dropout']}")
    print(f" -> larghezza canali :      {selected_config['channels']}")
    # print(f" -> Macro F1 medio in CV: {best_row['f1_mean']:.4f}")
    print(f" -> Macro F1 medio in CV:   {selected_config['f1_mean']:.4f} ± {selected_config['f1_std']:.4f}")
    print(f" -> ROC-AUC medio in CV :   {selected_config['auc_mean']:.4f} ± {selected_config['auc_std']:.4f}")

    # Analisi Comparativa della Preparazione dei Dati
    print("\n" + "="*71)
    print(" ANALISI COMPARATIVA DELLA PREPARAZIONE DEI DATI ")
    print("="*71)
    # print("[INFO] Avvio test: viene applicata la Z-Score globale includendo erroneamente le feature discrete...")
    print("[INFO] Utilizzando la configurazione vincente, il seguente test appica\n     la Z-Score in maniera globale (includendo anche le feature discrete).")
    
    f1_wrong, auc_wrong = tr.run_cross_validation(
        dataset=train_val_dataset, 
        model_class=best_model_class,
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.DEFAULT_BATCH_SIZE, 
        train_fold_fn=tr.train_single_fold,
        seed=cfg.FIXED_SEED,
        lr=selected_config['lr'],
        dropout_p=selected_config['dropout'],
        hidden_channels=cfg.DEFAULT_HIDDEN_CHANNELS,
        excluded_features=None,
        noise_level=cfg.NOISE_LEVEL
    )
    
    wrong_f1_mean = np.mean(f1_wrong)
    wrong_auc_mean = np.mean(auc_wrong)
    
    print("\n" + "-"*30 + " VERDETTO " + "-"*31)
    print(f" -> Z-Score Selettiva (Metodologia Corretta) - Macro F1 Medio = {selected_config['f1_mean']:.4f} e ROC-AUC Medio = {selected_config['auc_mean']:.4f}")
    print(f" -> Z-Score Globale (Corruzione Logica) - Macro F1 Medio = {wrong_f1_mean:.4f} e ROC-AUC Medio = {wrong_auc_mean:.4f}")
    # print(" Nota: La maschera è fondamentale per preservare la natura discreta di alcune feature (0/1).")
    print("-"*71)

    # Generazione e salvataggio del grafico comparativo della preparazione dati
    metrics_label = ['Macro F1', 'ROC-AUC']
    correct_scores = np.array([selected_config['f1_mean'], selected_config['auc_mean']], dtype=float)
    wrong_scores = np.array([wrong_f1_mean, wrong_auc_mean], dtype=float)
    x_axis = np.arange(len(metrics_label))

    plt.figure(figsize=(8, 5))
    plt.bar(x_axis - 0.2, correct_scores, 0.4, label='Z-Score Selettiva (Corretta)', color='teal')
    plt.bar(x_axis + 0.2, wrong_scores, 0.4, label='Z-Score Globale (Errata)', color='crimson')
    plt.xticks(x_axis, metrics_label)
    plt.ylabel("Punteggio Medio in CV")
    plt.title(f"Impatto della Preparazione dei Dati sulle Performance (Protein{selected_config.name})")

    # Adattamento dinamico dell'asse Y per massimizzare la leggibilità del confronto
    max_score = max(np.max(correct_scores), np.max(wrong_scores))
    plt.ylim(0, max_score + 0.05)
    
    plt.legend(loc="lower left")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    data_prep_plot_path = results_dir / "data_preparation_comparison.png"
    plt.savefig(data_prep_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n[INFO]: Salvataggio grafico associato al test...")

    # Analisi sensibilità architetturale
    print("\n" + "="*71)
    print(f" ANALISI SENSIBILITA' ARCHITETTURALE (hidden_channel: {cfg.ALT_HIDDEN_CHANNELS}) ")
    print("="*71)

    f1_spot, auc_spot = tr.run_cross_validation(
        dataset=train_val_dataset,
        model_class=best_model_class,
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.DEFAULT_BATCH_SIZE,
        train_fold_fn=tr.train_single_fold,
        seed=cfg.FIXED_SEED,
        lr=selected_config['lr'],
        dropout_p=selected_config['dropout'],
        hidden_channels=cfg.ALT_HIDDEN_CHANNELS,
        excluded_features=excluded_indexes,
        noise_level=cfg.NOISE_LEVEL
    )

    f1_spot_mean = np.mean(f1_spot)
    auc_spot_mean = np.mean(auc_spot)

    print("\n" + "-"*30 + " VERDETTO " + "-"*31)
    print(f"\n-> Risultato Test Attuale (Channels={cfg.ALT_HIDDEN_CHANNELS}): Macro F1 Medio = {f1_spot_mean:.4f} e ROC-AUC Medio = {auc_spot_mean:.4f}")
    print(f"-> Risultato Griglia Base (Channels={cfg.DEFAULT_HIDDEN_CHANNELS}): Macro F1 Medio = {selected_config.f1_mean:.4f} e ROC-AUC Medio = {selected_config.auc_mean:.4f}\n")

    best_config = selected_config.copy()

    # Selezione automatica basata sul verdetto dell'analisi
    if f1_spot_mean > selected_config.f1_mean:
        best_config["channels"] = cfg.ALT_HIDDEN_CHANNELS
        print(f"\n [STRUTTURA AGGIORNATA]: Scelti {cfg.ALT_HIDDEN_CHANNELS} canali per l'addestramento finale.")
    else:
        print(f"\n [STRUTTURA CONFERMATA]: Mantenuti i {cfg.DEFAULT_HIDDEN_CHANNELS} canali di default.")

    print("-"*71)

    # Addestramento definitivo con configurazione migliore
    print("\n" + "="*71)
    print("AVVIO ADDESTRAMENTO FINALE")
    print("="*71)
    print("\n[INFO]: Il processo di addestramento implementa la Noise Injection\n    e sfrutta i parametri della configurazione migliore:")

    df_hyperparams = best_config.drop(["f1_mean", "f1_std", "auc_mean", "auc_std"]).to_frame().T

    # Stampa su terminale
    with pd.option_context('display.max_columns', None, 'display.width', 1000):
        print(df_hyperparams.round(4).to_string(col_space=12, index=False) + "\n")

    # Addestramento finale del modello (gestione interna sia di Z-Score che Noise Injection)
    final_model, history_loss, final_mean, final_std = tr.train_final_model(
        dataset=train_val_dataset, 
        model_class=best_model_class,
        lr=best_config['lr'],
        dropout_p=best_config['dropout'],
        hidden_channels=best_config['channels'],
        epochs=cfg.DEFAULT_EPOCHS,
        batch_size=cfg.DEFAULT_BATCH_SIZE,
        noise_level=cfg.NOISE_LEVEL,
        excluded_features=excluded_indexes
    )

    # Valutazione sul Test Set
    print("\n" + "="*71)
    print(" VALUTAZIONE SUL TEST SET (DATI MAI VISTI) ")
    print("="*71)
    test_dataset_normalized = dl.apply_z_score(test_dataset, final_mean, final_std, excluded_features=excluded_indexes)
    test_loader = DataLoader(test_dataset_normalized, batch_size=cfg.DEFAULT_BATCH_SIZE, shuffle=False, drop_last=False)

    # Valutazione test set
    y_test_true, y_test_pred, y_test_score = tr.evaluate_model(final_model, test_loader)

    # Calcolo Metriche Definitive
    acc = accuracy_score(y_test_true, y_test_pred)
    f1_macro = f1_score(y_test_true, y_test_pred, average="macro")
    precision = precision_score(y_test_true, y_test_pred, zero_division=0)
    recall = recall_score(y_test_true, y_test_pred)
    roc_auc = roc_auc_score(y_test_true, y_test_score)

    # Esportazione delle metriche definitive del Test Set 
    final_metrics = pd.DataFrame({
        "Accuracy": [acc],
        "Macro_F1": [f1_macro],
        "Precision": [precision],
        "Recall": [recall],
        "ROC_AUC": [roc_auc]
    })

    final_metrics.to_csv(
        results_dir / "final_test_metrics.csv", index=False, float_format="%.4f"
    )

    # print(f" - Accuracy  : {acc:.4f}")
    # print(f" - Macro F1  : {f1_macro:.4f}  <-- Metrica di riferimento finale")
    # print(f" - Precision : {precision:.4f} (Capacità di evitare falsi positivi)")
    # print(f" - Recall    : {recall:.4f} (Capacità di individuare gli enzimi)")
    # print(f" - ROC-AUC   : {roc_auc:.4f}")
    # print("-"*71)

    # print("=============================================================")
    # print(" METRICHE DEFINITIVE SUL TEST SET (MODELLO VINCENTE) ")
    # print("=============================================================")
    print(f" - Accuracy  : {acc:.4f}")
    print(f" - Macro F1  : {f1_macro:.4f}  <-- Metrica di riferimento")
    print(f" - Precision : {precision:.4f}  (Capacità di evitare falsi positivi)")
    print(f" - Recall    : {recall:.4f}  (Capacità di scovare tutti gli enzimi)")
    print(f" - ROC-AUC   : {roc_auc:.4f}")
    print("=============================================================\n")

    # 7. GENERAZIONE E SALVATAGGIO DEI GRAFICI FINALI
    print("[INFO] Salvataggio dei grafici di valutazione finali...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Loss
    ax[0].plot(range(1, len(history_loss) + 1), history_loss, label='Training Loss', color='teal', linewidth=2)
    ax[0].set_title('Curva di Apprendimento (Final Training)')
    ax[0].set_xlabel('Epoca')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True, linestyle='--', alpha=0.6)
    ax[0].legend()

    # Plot Matrice di Confusione
    cm = confusion_matrix(y_test_true, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1], cbar=False,
                xticklabels=['Non-Enzima', 'Enzima'], yticklabels=['Non-Enzima', 'Enzima'])
    ax[1].set_title('Matrice di Confusione sul Test Set')
    ax[1].set_xlabel('Predetto')
    ax[1].set_ylabel('Reale')

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test_true, y_test_score)
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
    final_plots_path = results_dir / "final_evaluation_plots.png"
    plt.savefig(final_plots_path, dpi=300)
    plt.close()
    
    print(f"[OK] Tutti gli artefatti d'esame sono stati correttamente esportati nella cartella:\n     {results_dir.resolve()}")
    print("="*71 + "\n")

if __name__ == "__main__":
    main()