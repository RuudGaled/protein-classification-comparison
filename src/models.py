import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from src import config as cfg

class ProteinGCN(nn.Module):
    """
    Graph Convolutional Network (GCN) per la classificazione binaria di grafi proteici.
    Utilizza il message passing standard con aggregazione normalizzata dei nodi vicini.
    """
    def __init__(self, in_channels: int = 32, hidden_channels: int = cfg.DEFAULT_HIDDEN_CHANNELS, dropout_p: float = 0.3):
        super(ProteinGCN, self).__init__()
        
        # Livelli convolutivi sul grafo
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Classificatore finale (Simple MLP con Dropout)
        # La concatenazione di Global Mean Pooling e Global Max Pooling
        # produce un vettore di dimensione hidden_channels * 2.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_channels // 2, 1)  # Restituisce un logit per BCEWithLogitsLoss.
        )

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Primo livello convolutivo + funzione di attivazione.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Secondo livello convolutivo + funzione di attivazione.
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Fase Readout: Concatenazione di Global Mean e Max Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=-1) 
        
        # Classificazione del grafo
        return self.classifier(x_pooled)


class ProteinGAT(nn.Module):
    """
    Graph Attention Network (GAT) per la classificazione binaria di grafi proteici.
    Utilizza meccanismi di attenzione per pesare il contributo dei nodi vicini.
    """
    def __init__(self, in_channels: int = 32, hidden_channels: int = cfg.DEFAULT_HIDDEN_CHANNELS, dropout_p: float = 0.3):
        super(ProteinGAT, self).__init__()
        
        # Livelli convolutivi con meccanismo di attenzione
        # Layer 1: 2 teste di attenzione.
        self.conv1 = GATConv(in_channels, hidden_channels // 2, heads=2, concat=True)
        
        # Layer 2: una sola testa per ottenere direttamente
        # una rappresentazione di dimensione hidden_channels.
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=True)
        
        # Classificatore finale (identico a GCN)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Primo livello di attenzione + funzione di attivazione.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Secondo livello di attenzione + funzione di attivazione.
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Fase Readout: Concatenazione di Global Mean e Max Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=-1)
        
        # Classificazione del grafo.
        return self.classifier(x_pooled)