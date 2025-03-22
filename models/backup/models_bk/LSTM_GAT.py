import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_config import Lstm_gat_Config
from models import LSTM_GAT_Model
from train_eval import run_experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    from run_mymodel import train_data, val_data, test_data
    
    model = LSTM_GAT_Model(
        input_size=Lstm_gat_Config.input_size,
        hidden_size=Lstm_gat_Config.hidden_size,
        output_size=Lstm_gat_Config.output_size,
        num_layers=Lstm_gat_Config.num_layers,
        num_heads=Lstm_gat_Config.num_heads,
        dropout=Lstm_gat_Config.dropout
    ).to(device)
    
    run_experiment(model, train_data, val_data, test_data, Lstm_gat_Config, "LSTM_GAT")