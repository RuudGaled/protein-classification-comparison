import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

class ProteinGCN(nn.Module):
    """
    Graph Convolutional Network (GCN) per la classificazione binaria di grafi proteici.
    Sfrutta il message passing standard con normalizzazione del vicinato.
    """
    def __init__(self, in_channels: int = 32, hidden_channels: int = 64, dropout_p: float = 0.3):
        super(ProteinGCN, self).__init__()
        
        # --- Strati di Convoluzione su Grafo ---
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # --- Classificatore Finale (Simple MLP con Dropout) ---
        # Poiché concateniamo Mean e Max Pooling, l'input del fully connected raddoppia
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_channels // 2, 1)  # 1 output per il logit (BCEWithLogitsLoss)
        )

    def forward(self, data):
        # x: [num_nodi, in_channels], edge_index: [2, num_archi], batch: [num_nodi]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Primo livello convolutivo + attivazione
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2. Secondo livello convolutivo
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 3. Readout Phase: Concatenazione di Global Mean e Max Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=-1) # [batch_size, hidden_channels * 2]
        
        # 4. Classificazione
        return self.classifier(x_pooled)


class ProteinGAT(nn.Module):
    """
    Graph Attention Network (GAT) per la classificazione binaria di grafi proteici.
    Sfrutta meccanismi di attenzione asimmetrica sui nodi vicini.
    """
    def __init__(self, in_channels: int = 32, hidden_channels: int = 64, dropout_p: float = 0.3):
        super(ProteinGAT, self).__init__()
        
        # --- Strati di Convoluzione ad Attenzione ---
        # Layer 1: usiamo 2 teste. L'output concatenato sarà di dimensione (hidden_channels // 2) * 2 = hidden_channels
        self.conv1 = GATConv(in_channels, hidden_channels // 2, heads=2, concat=True)
        
        # Layer 2: usiamo 1 sola testa per stabilizzare l'output a hidden_channels prima del pooling
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=True)
        
        # --- Classificatore Finale (Identico a GCN per equità di confronto) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Primo livello di attenzione + attivazione
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2. Secondo livello di attenzione
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 3. Readout Phase: Concatenazione di Global Mean e Max Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=-1)
        
        # 4. Classificazione
        return self.classifier(x_pooled)