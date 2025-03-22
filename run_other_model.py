import torch
from models.models import (
    GATModel, 
    GRUModel, 
    GRU_GAT_Model, 
    LSTMModel, 
    LSTM_GAT_Model, 
    RNNModel
)
from data.data_util import get_train_val_test_data
from train_utils.train_eval import run_experiment
from models.model_config import (
    Base_Config,
    Gat_Config,
    Gru_Config,
    Gru_gat_Config,
    Lstm_Config,
    Lstm_gat_Config,
    Rnn_Config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(model_class, config):
    """创建模型实例"""
    if model_class in [GRUModel, LSTMModel, RNNModel]:
        model = model_class(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_layers=config.num_layers
        )
    else:  # GAT, GRU_GAT, LSTM_GAT
        model = model_class(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    return model.to(device)

def run_all_models():
    """运行所有模型"""
    models = {
        'GAT': (GATModel, Gat_Config),
        'GRU': (GRUModel, Gru_Config),
        'GRU_GAT': (GRU_GAT_Model, Gru_gat_Config),
        'LSTM': (LSTMModel, Lstm_Config),
        'LSTM_GAT': (LSTM_GAT_Model, Lstm_gat_Config),
        'RNN': (RNNModel, Rnn_Config)
    }
    
    # 获取数据（只需获取一次）
    train_data, val_data, test_data = get_train_val_test_data(Base_Config)
    
    results = {}
    for model_name, (model_class, model_config) in models.items():
        print(f"\nRunning {model_name} model...")
        model = create_model(model_class, model_config)
        results[model_name] = run_experiment(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=model_config,
            model_name=model_name
        )
    
    return results

if __name__ == "__main__":
    print("Starting model training and evaluation...")
    
    # 运行所有模型
    results = run_all_models()
    
    # 打印所有结果
    print("\nAll models results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
