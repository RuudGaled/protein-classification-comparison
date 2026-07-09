# Progetto di Deep Learning: Classificazione Funzionale di Strutture Proteiche

Questo repository contiene una pipeline ingegnerizzata di Deep Learning per la classificazione binaria di grafi molecolari (enzimi vs non-enzimi). Il progetto implementa reti neurali su grafi (GCN e GAT) e include sperimentazioni avanzate di data preparation (Log-Transform, Z-Score selettiva, Noise Injection) e studi di ridondanza delle feature (rimozione one-hot).

**Nota sulla Riproducibilità:** Per garantire la massima riproducibilità tra macchine diverse (Windows/Mac/Linux), l'intero progetto e le relative dipendenze sono stati configurati esplicitamente in modalità **CPU-Only**.

---

## Struttura del Progetto

```text
protein-classification-comparison/
│
├── data/                  # Cartella per il dataset originale
│   └── dataset.pt         # File PyTorch (non incluso nel repo)
│
├── src/                   # Codice sorgente modulare
│   ├── __init__.py        
│   ├── data_loader.py     # Pulizia dati, trasformazioni (Z-Score, Log-Transform, Noise)
│   ├── models.py          # Architettura delle reti (ProteinGCN, ProteinGAT)
│   ├── train.py           # Pipeline di Cross-Validation e Training finale
│   ├── utils.py           # Funzioni di utilità, plot grafici, seed
│   └── config.py          # Variabili globali e percorsi
│
├── results/               # Generata dinamicamente durante l'esecuzione
│
├── main.py                # Script di esecuzione totale automatizzata
│
├── notebooks/
│   └── model_comparison.ipynb  # Notebook interattivo per l'analisi esplorativa
│
├── requirements.txt       # Dipendenze Python blindate (pip)
└── environment.yml        # Configurazione ambiente Conda
```
---

## 1. Setup dell'Ambiente di Sviluppo

Per riprodurre gli esperimenti, è necessario replicare l'ambiente virtuale isolato. Di seguito le istruzioni sia per utenti standard (`venv`) che per utenti Conda.

### Opzione A: Utilizzando `venv` (Senza Conda)

> [!WARNING]
> **Prerequisito fondamentale**: È richiesta l'installazione nativa di **Python 3.11** nel sistema per garantire la compatibilità con le librerie di Deep Learning.
>  Se non presente sul proprio sistema, è necessario installarlo prima di eseguire i comandi sottostanti.

**1. Creazione ambiente virtuale**

È possibile generare l'ambiente virtuale tramite il tool nativo di Python (forzando specificamente la versione 3.11 per evitare conflitti):

```bash
# Su WINDOWS
py -3.11 -m venv dl_env

# oppure, se il launcher `py` non è disponibile ma l'eseguibile di Python 3.11 è nel PATH:
python3.11 -m venv dl_env

# Su MAC/LINUX
python3.11 -m venv dl_env
```
**2. Attivazione dell'ambiente virtuale**

```bash
# Su WINDOWS
dl_env\Scripts\activate

# Su MAC/LINUX
source dl_env/bin/activate
```
**3. Aggiornamento pip e installazione librerie**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**4. (Opzionale)Setup per l'esecuzione del Notebook**
Se si desidera eseguire il progetto tramite Jupyter Notebook (`jupyter lab`) e l'ambiente virtuale è stato generato tramite `venv`, è necessario installare manualmente i pacchetti per l'interfaccia grafica:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name=dl_env --display-name "Python (dl_env)"
```

### Opzione B: Utilizzando Conda / Miniconda

**1. Creazione dell'ambiente Conda a partire dal file fornito** (installerà automaticamente Python 3.11 e le dipendenze di sistema Jupyter):

```bash
conda env create -f environment.yml
```
**2. Attivazione dell'ambiente** (il nome `dl_env` è vincolato dal file yaml)

```bash
conda activate dl_env
```
**3. Installazione delle librerie Deep Learning strettamente versionate:**
   
```bash
python -m pip install -r requirements.txt
```
**4. (Opzionale per utenti Notebook) Collega l'ambiente a Jupyter:**

```bash
python -m ipykernel install --user --name=dl_env --display-name "Python (dl_env)"
```

## 2. Preparazione dei Dati

Il **Dataset G3: "Classificazione Funzionale di Strutture Proteiche"** non è tracciato su Git (ignorato tramite `.gitignore`). Per poter eseguire i modelli, è necessario inserire lo specifico file `dataset.pt` all'interno della directory `data/`.

Il percorso atteso dallo script sarà dunque: `protein-classification-comparison/data/dataset.pt`.

## 3. Riproduzione degli Esperimenti e Training

L'intera pipeline (Grid Search, Esperimenti di Data Preparation, Rimozione One-Hot e Addestramento Finale) è completamente automatizzata.

È possibile eseguire il progetto in due modi:

### Metodo CLI (Terminale)

Solo dopo aver attivato l'ambiente virtuale `dl_env`, dalla root del progetto, eseguire:

```bash
python main.py
```

**Suggerimento**: È consigliato, prima di lanciare lo script, ingrandire la finestra del terminale (preferibilmente a schermo intero). Lo script stampa a schermo tabelle ASCII e report progressivi che richiedono spazio orizzontale per una formattazione ottimale e una facile lettura.

### Metodo Jupyter Notebook

Se si preferisce un'esecuzione step-by-step accompagnata da analisi visuale interattiva:

1. Lanciare il server Jupyter: `jupyter lab`
2. Navigare nella cartella `notebooks/` e aprire il file `model_comparison.ipynb`.
3. Assicurarsi che il kernel selezionato in alto a destra sia quello creato in precedenza (`dl_env`).
4. Eseguire le celle sequenzialmente.

## 4. Risultati e Artefatti

Anche se la cartella `results/` è presente nella struttura della repository, il suo contenuto è ignorato da Git. L'esecuzione del codice di progetto (tramite script `main.py` o Notebook) creerà e popolerà automaticamente questa cartella.

Al suo interno verranno generati i seguenti artefatti utili alla validazione degli esperimenti:

- `node_features_distributions.png`: Distribuzione statistica (Istogramma + Box Plot) delle prime quattro feature che presentano un'elevata asimmetria, a giustificazione dell'utilizzo della Log-Transform.
- `grid_search_results.csv`: Tabella riassuntiva con le performance (Macro F1 e AUC) di tutte le configurazioni testate nella Grid Search iniziale.
- `test_log_transform.png`: Grafico comparativo delle performance (Baseline vs Log-Transform).
- `test_no_onehot.png`: Grafico comparativo delle performance (Baseline vs Rimozione delle feature One-Hot).
- `test_alt_hidden_channels.png`: Grafico comparativo delle performance variando la dimensionalità dei layer nascosti.
- `final_test_metrics.csv`: Metriche definitive (Accuracy, F1, Precision, Recall, AUC) ottenute dal modello finale valutato sul Test Set.
- `final_evaluation_plots.png`: Dashboard visuale finale contenente la Curva di Apprendimento (Loss), la Matrice di Confusione e la Curva ROC.