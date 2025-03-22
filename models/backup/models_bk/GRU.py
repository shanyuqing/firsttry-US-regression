import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_config import Gru_Config
from models import GRUModel
from train_eval import run_experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    from run_mymodel import train_data, val_data, test_data
    
    model = GRUModel(
        input_size=Gru_Config.input_size,
        hidden_size=Gru_Config.hidden_size,
        output_size=Gru_Config.output_size,
        num_layers=Gru_Config.num_layers
    ).to(device)
    
    run_experiment(model, train_data, val_data, test_data, Gru_Config, "GRU")