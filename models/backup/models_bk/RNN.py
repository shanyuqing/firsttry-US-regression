import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_config import Rnn_Config
from models import RNNModel
from train_eval import run_experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    from run_mymodel import train_data, val_data, test_data
    
    model = RNNModel(
        input_size=Rnn_Config.input_size,
        hidden_size=Rnn_Config.hidden_size,
        output_size=Rnn_Config.output_size,
        num_layers=Rnn_Config.num_layers
    ).to(device)
    
    run_experiment(model, train_data, val_data, test_data, Rnn_Config, "RNN")

# test mse_loss: 0.0011950434874896869
# test mae_loss: 0.023496972193686187
# test rmse_loss: 0.03231902296659608
# test mape_loss: 5.0151067022139655
# rnn_IC: 0.0054869267238314805
# rnn_RankIC: 0.017896397745951354
# rnn_ICIR: 0.1698794370098497
# rnn_RankICIR: 2.0183863890314386e-06