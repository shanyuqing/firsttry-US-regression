import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, dropout):
        super(GATModel, self).__init__()
        
        # 第一层GAT：多注意力头+拼接
        self.gat1 = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # 层间Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 第二层GAT：单注意力头（输出层）+平均
        self.gat2 = GATConv(
            in_channels=hidden_size * num_heads,
            out_channels=output_size,
            heads=1,
            dropout=dropout,
            concat=False
        )

    def forward(self, x, edge_index):
        # 第一层
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.gat2(x, edge_index)
        return x

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

class GRU_GAT_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super(GRU_GAT_Model, self).__init__()
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
            dropout=dropout
        )
        self.gat2 = GATConv(
            in_channels=hidden_size * num_heads,
            out_channels=output_size,
            heads=1,
            dropout=dropout
        )

    def forward(self, x, edge_index):
        # First, process through GRU (Time-series part)
        x, _ = self.gru(x)
       
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class LSTM_GAT_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout=0.5):
        super(LSTM_GAT_Model, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index):
        # First, process through LSTM
        x, _ = self.lstm(x)
        
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out 