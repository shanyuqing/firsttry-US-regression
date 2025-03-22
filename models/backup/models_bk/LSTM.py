import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_config import Lstm_Config
from models import LSTMModel
from train_eval import run_experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    from run_mymodel import train_data, val_data, test_data
    
    model = LSTMModel(
        input_size=Lstm_Config.input_size,
        hidden_size=Lstm_Config.hidden_size,
        output_size=Lstm_Config.output_size,
        num_layers=Lstm_Config.num_layers
    ).to(device)
    
    run_experiment(model, train_data, val_data, test_data, Lstm_Config, "LSTM")

# test mse_loss: 0.003601340463545722
# test mae_loss: 0.048810025043596474
# test rmse_loss: 0.058897479063817086
# test mape_loss: 10.162018037689519
# lstm_IC: 0.012928597411330634
# lstm_RankIC: 0.03265764053898403
# lstm_ICIR: 0.3329360315659576
# lstm_RankICIR: 3.683581732688043e-06