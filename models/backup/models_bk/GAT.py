import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np 
import torch.nn.functional as F
from scipy.stats import spearmanr  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_mymodel import train_data, test_data, val_data
from model_config import Gat_Config
from baseline.baseline_model import calculate_all_metrics, train_model, evaluate_model, test_model
from models import GATModel
from train_eval import run_experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            heads=num_heads,         # 使用num_heads个注意力头
            dropout=dropout,
            concat=True               # 默认True，拼接多头结果
        )
        
        # 层间Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 第二层GAT：单注意力头（输出层）+平均
        self.gat2 = GATConv(
            in_channels=hidden_size * num_heads,  # 输入维度是第一层的输出
            out_channels=output_size,
            heads=1,                # 输出层使用单头
            dropout=dropout,
            concat=False             # 不拼接（对多头结果取平均）
        )

    def forward(self, x, edge_index):
        # 第一层
        x = self.gat1(x, edge_index)
        x = F.elu(x)                 # 更常用的GAT激活函数
        x = self.dropout(x)
        
        # 第二层
        x = self.gat2(x, edge_index)
        return x  
    
if __name__ == "__main__":
    from run_mymodel import train_data, val_data, test_data
    from model_config import Gat_Config
    from train_eval import run_experiment
    from models import GATModel
    
    model = GATModel(
        input_size=Gat_Config.input_size,
        hidden_size=Gat_Config.hidden_size,
        output_size=Gat_Config.output_size,
        num_heads=Gat_Config.num_heads,
        dropout=Gat_Config.dropout
    ).to(device)
    
    run_experiment(model, train_data, val_data, test_data, Gat_Config, "GAT")

    


