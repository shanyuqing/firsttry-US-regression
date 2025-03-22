import os
import sys
import torch
import torch.optim as optim
import optuna

# 修改系统路径以正确导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从当前目录导入相关模块
from train_eval import train_model  # 改用新的 train_eval.py
from models import (
    GATModel, 
    GRUModel, 
    GRU_GAT_Model, 
    LSTMModel, 
    LSTM_GAT_Model, 
    RNNModel
)

# 从项目根目录导入数据
from data.data_util import get_train_val_test_data
from config import model_config  # 假设你有一个配置文件

# 获取数据
train_data, val_data, test_data = get_train_val_test_data(model_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_objective(model_class, input_size, output_size):
    """
    创建用于特定模型的目标函数
    """
    def objective(trial):
        # 基础超参数搜索空间
        base_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "hidden_size": trial.suggest_int("hidden_size", 64, 512),
            "optimizer_name": trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
            "epochs": trial.suggest_int("epochs", 10, 256)
        }

        # 根据模型类型添加特定参数
        model_params = base_params.copy()
        
        if model_class in [GATModel, GRU_GAT_Model, LSTM_GAT_Model]:
            # GAT相关模型的特定参数
            model_params.update({
                "num_heads": trial.suggest_int("num_heads", 1, 6),
                "dropout": trial.suggest_float("dropout", 0.1, 0.9),
                "num_layers": trial.suggest_int("num_layers", 1, 6)
            })
        elif model_class in [GRUModel, LSTMModel, RNNModel]:
            # RNN系列模型的特定参数
            model_params.update({
                "num_layers": trial.suggest_int("num_layers", 1, 6)
            })

        # 创建模型实例
        if model_class in [GRUModel, LSTMModel, RNNModel]:
            model = model_class(
                input_size=input_size,
                hidden_size=model_params["hidden_size"],
                output_size=output_size,
                num_layers=model_params["num_layers"]
            ).to(device)
        else:  # GAT相关模型
            model = model_class(
                input_size=input_size,
                hidden_size=model_params["hidden_size"],
                output_size=output_size,
                num_layers=model_params["num_layers"],
                num_heads=model_params["num_heads"],
                dropout=model_params["dropout"]
            ).to(device)

        # 定义优化器
        if model_params["optimizer_name"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"], weight_decay=5e-3)
        else:
            optimizer = optim.SGD(model.parameters(), lr=model_params["learning_rate"], weight_decay=5e-3)

        # 训练模型
        loss_values, val_loss_values, _ = train_model(
            model, train_data, val_data, 
            epochs=model_params["epochs"], 
            lr=model_params["learning_rate"]
        )

        # 返回验证集损失的最小值
        return min(val_loss_values)

    return objective

def tune_model(model_class, model_name, input_size=18, output_size=1, n_trials=50):
    """
    对指定模型进行超参数调优
    
    Args:
        model_class: 模型类
        model_name: 模型名称
        input_size: 输入维度
        output_size: 输出维度
        n_trials: 调优次数
    """
    print(f"\nTuning {model_name}...")
    study = optuna.create_study(direction="minimize")
    objective = create_objective(model_class, input_size, output_size)
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{model_name} Best trial:")
    trial = study.best_trial
    print(f"  Value (Validation Loss): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.params

def tune_all_models(n_trials=50):
    """
    对所有模型进行超参数调优
    """
    models = {
        'GAT': GATModel,
        'GRU': GRUModel,
        'GRU_GAT': GRU_GAT_Model,
        'LSTM': LSTMModel,
        'LSTM_GAT': LSTM_GAT_Model,
        'RNN': RNNModel
    }
    
    best_params = {}
    for model_name, model_class in models.items():
        best_params[model_name] = tune_model(model_class, model_name, n_trials=n_trials)
    
    return best_params

if __name__ == "__main__":
    # 运行所有模型的调优
    best_params = tune_all_models()
    
    # 保存最佳参数
    print("\nBest parameters for all models:")
    for model_name, params in best_params.items():
        print(f"\n{model_name}:")
        for param_name, value in params.items():
            print(f"    {param_name}: {value}") 