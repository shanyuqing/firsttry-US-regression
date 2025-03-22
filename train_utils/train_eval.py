import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import spearmanr
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import GATModel, GRUModel, GRU_GAT_Model, LSTMModel, LSTM_GAT_Model, RNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_corrcoef(x, y):
    """安全的相关系数计算"""
    with np.errstate(invalid='ignore'):
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return corr if not np.isnan(corr) else 0.0

def safe_spearmanr(x, y):
    """安全的Spearman相关系数计算"""
    with np.errstate(invalid='ignore'):
        if len(x) < 2 or len(y) < 2:
            return 0.0
        corr = spearmanr(x, y).correlation
        return corr if not np.isnan(corr) else 0.0

def calculate_ic(pred, target):
    """计算信息系数 (Information Coefficient)"""
    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    return np.corrcoef(pred, target)[0, 1]

def calculate_metrics(preds, targets):
    """计算所有相关指标"""
    ic = safe_corrcoef(preds, targets)
    rank_ic = safe_spearmanr(preds, targets)
    
    # ICIR计算
    ic_series = []
    for i in range(1, len(preds)):
        ic_series.append(np.corrcoef(preds[:i], targets[:i])[0,1])
    icir = np.nanmean(ic_series) / (np.nanstd(ic_series) + 1e-8) if len(ic_series) > 0 else 0
    
    # Rank ICIR计算
    rank_ic_series = []
    for i in range(1, len(preds)):
        rank_ic_series.append(spearmanr(preds[:i], targets[:i]).correlation)
    rank_icir = np.nanmean(rank_ic_series) / (np.nanstd(rank_ic_series) + 1e-8) if len(rank_ic_series) > 0 else 0
    
    return ic, rank_ic, icir, rank_icir

def train_model(model, train_data, val_data, epochs, lr):
    """训练函数"""
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_metrics': [],
        'train_ic': []
    }

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_metrics = {'total_loss': 0.0, 'total_ic': 0.0, 'num_samples': 0}
        
        for batch in train_data:
            x, edge_index, y = batch[0], batch[1], batch[3]
            optimizer.zero_grad()
            
            y_pred = model(x, edge_index)
            y = y.view(-1, 1)
            
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            
            batch_size = y.size(0)
            epoch_metrics['total_loss'] += loss.item() * batch_size
            epoch_metrics['num_samples'] += batch_size
            
            with torch.no_grad():
                batch_ic = calculate_ic(y_pred, y)
                epoch_metrics['total_ic'] += batch_ic * batch_size

        avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_samples']
        avg_ic = epoch_metrics['total_ic'] / epoch_metrics['num_samples']
        history['train_loss'].append(avg_loss)
        history['train_ic'].append(avg_ic)
        
        # 验证集评估
        val_metrics = evaluate_model(model, val_data)
        history['val_metrics'].append(val_metrics)
        
        scheduler.step(val_metrics['val_mse'])
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Train IC: {avg_ic:.4f} | "
              f"Val MSE: {val_metrics['val_mse']:.4f} | "
              f"Val MAE: {val_metrics['val_mae']:.4f} | "
              f"Val MAPE: {val_metrics['val_mape']:.2f}%")
    
    return history['train_loss'], history['val_metrics'], history['train_ic']

def evaluate_model(model, data):
    """评估函数，用于验证和测试"""
    model.eval()
    total = {'mse': 0.0, 'mae': 0.0, 'mape': 0.0, 'samples': 0}
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in data:
            x, edge_index, y = batch[0], batch[1], batch[3]
            y_pred = model(x, edge_index)
            y = y.view(-1, 1)
            
            all_preds.append(y_pred)
            all_targets.append(y)
            
            total['mse'] += F.mse_loss(y_pred, y, reduction='sum').item()
            total['mae'] += F.l1_loss(y_pred, y, reduction='sum').item()
            total['samples'] += y.size(0)
            
            non_zero_mask = y != 0
            if non_zero_mask.any():
                total['mape'] += torch.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / 
                                        y[non_zero_mask]).sum().item() * 100

    metrics = {
        'mse': total['mse'] / total['samples'],
        'mae': total['mae'] / total['samples'],
        'rmse': torch.sqrt(torch.tensor(total['mse'] / total['samples'])).item(),
        'mape': (total['mape'] / total['samples']) if total['mape'] > 0 else float('nan')
    }
    
    # 计算IC相关指标
    preds = torch.cat(all_preds).cpu().numpy()
    targets = torch.cat(all_targets).cpu().numpy()
    ic, rank_ic, icir, rank_icir = calculate_metrics(preds, targets)
    metrics.update({
        'ic': ic,
        'rank_ic': rank_ic,
        'icir': icir,
        'rank_icir': rank_icir
    })
    
    return metrics

def run_experiment(model, train_data, val_data, test_data, config, model_name):
    """运行单个模型的实验"""
    # 训练模型
    loss_values, val_metrics, ic_values = train_model(
        model, train_data, val_data, 
        epochs=config.epochs, lr=config.lr
    )

    # 绘制训练曲线
    plt.figure(figsize=(8, 6))
    epochs_range = range(config.epochs)
    plt.plot(epochs_range, loss_values, marker='o', linestyle='-', color='b', label='train_loss')
    plt.plot(epochs_range, [m['mse'] for m in val_metrics], marker='o', linestyle='-', color='g', label='val_loss')
    plt.plot(epochs_range, ic_values, marker='o', linestyle='-', color='r', label='ic')
    plt.title('Loss/IC Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss/IC', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{model_name}_training.png')
    plt.close()

    # 测试评估
    test_metrics = evaluate_model(model, test_data)
    
    # 打印测试结果
    print(f"\n{model_name} Test Results:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.6f}")

    return test_metrics

def create_model(model_class, config):
    """
    根据配置创建模型实例
    """
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
            num_layers=config.num_layers if hasattr(config, 'num_layers') else None,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    return model.to(device)

def run_all_experiments(train_data, val_data, test_data, configs):
    """
    运行所有模型的实验
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        configs: 包含所有模型配置的字典
    """
    models = {
        'GAT': (GATModel, configs.Gat_Config),
        'GRU': (GRUModel, configs.Gru_Config),
        'GRU_GAT': (GRU_GAT_Model, configs.Gru_gat_Config),
        'LSTM': (LSTMModel, configs.Lstm_Config),
        'LSTM_GAT': (LSTM_GAT_Model, configs.Lstm_gat_Config),
        'RNN': (RNNModel, configs.Rnn_Config)
    }

    results = {}
    for model_name, (model_class, config) in models.items():
        print(f"\nRunning experiment for {model_name}...")
        model = create_model(model_class, config)
        results[model_name] = run_experiment(model, train_data, val_data, test_data, config, model_name)

    return results

if __name__ == "__main__":
    from config import model_config
    from data.data_util import get_train_val_test_data
    
    # 获取数据
    train_data, val_data, test_data = get_train_val_test_data(model_config)
    
    # 运行所有实验
    results = run_all_experiments(train_data, val_data, test_data, model_config)
    
    # 打印所有结果
    print("\nAll experiments results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}") 