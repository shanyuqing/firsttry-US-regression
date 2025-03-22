import os
import pandas as pd
import numpy as np
import torch
from data.generate_data import normalize

def load_stock_data(folder_path, n_company, num_days, num_features):
    """
    加载股票数据
    
    Args:
        folder_path: 数据文件夹路径
        n_company: 公司数量
        num_days: 天数
        num_features: 特征数量
        
    Returns:
        normalized_data: 归一化后的股票数据
    """
    # 获取所有txt文件
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    all_data = pd.DataFrame()

    # 读取并合并所有文件数据
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, sep='\t', header=None, 
                        names=['dt', 'open', 'high', 'low', 'close', 'adj close', 'volume'])
        df = df.iloc[::-1].reset_index(drop=True)
        df['code'] = file_name[0:-4]
        all_data = pd.concat([all_data, df], ignore_index=True)

    # 转换为三维数组
    stock_data = np.zeros((n_company, num_days, num_features))
    for i, stock_code in enumerate(all_data['code'].unique()):
        stock_df = all_data[all_data['code'] == stock_code]
        stock_data[i] = stock_df.iloc[:, 1:-1].values

    # 数据归一化
    normalized_data = np.zeros_like(stock_data)
    for features_index in range(stock_data.shape[2]):
        stock_type_data = stock_data[:, :, features_index]
        global_min = np.min(stock_type_data)
        global_max = np.max(stock_type_data)
        normalized_data[:, :, features_index] = (stock_type_data - global_min) / (global_max - global_min)

    return normalized_data

def load_correlation_matrix(correlation_path, n_company, threshold=0.7):
    """
    加载相关性矩阵
    
    Args:
        correlation_path: 相关性矩阵文件路径
        n_company: 公司数量
        threshold: 相关性阈值
        
    Returns:
        sadj: 处理后的相关性矩阵
    """
    correlation_matrix = pd.read_csv(correlation_path)
    sadj = correlation_matrix.iloc[0:n_company, 1:(n_company+1)].values
    
    # 应用阈值
    sadj[abs(sadj) < threshold] = 0
    sadj = torch.tensor(sadj, dtype=torch.float32)
    
    # 对称化处理
    sadj = sadj + sadj.T.mul(sadj.T > sadj) - sadj.mul(sadj.T > sadj)
    
    # 归一化处理
    sadj = normalize(sadj + torch.eye(sadj.size(0), dtype=torch.float32))
    
    return torch.from_numpy(sadj)

def load_and_process_data(config):
    """
    加载并处理所有数据
    
    Args:
        config: 包含数据处理参数的配置对象
        
    Returns:
        stock_data: 处理后的股票数据
        sadj: 处理后的相关性矩阵
    """
    # 加载股票数据
    stock_data = load_stock_data(
        folder_path=config.data_path,
        n_company=config.n_company,
        num_days=config.num_days,
        num_features=config.num_features
    )
    
    # 加载相关性矩阵
    sadj = load_correlation_matrix(
        correlation_path=config.correlation_path,
        n_company=config.n_company,
        threshold=config.correlation_threshold
    )
    
    return stock_data, sadj

def get_train_val_test_data(config):
    """
    获取训练、验证和测试数据集
    
    Args:
        config: 配置对象，包含数据处理相关参数
        
    Returns:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
    """
    # 加载和处理数据
    stock_data, sadj = load_and_process_data(config)
    
    # 创建数据集
    from data.generate_data import create_data
    from sklearn.model_selection import train_test_split
    
    batch_data_list = create_data(
        stock_data=stock_data, 
        sadj=sadj, 
        time_window=config.time_window
    )
    
    # 划分数据集
    train_data, temp_data = train_test_split(
        batch_data_list, 
        test_size=0.4, 
        random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42
    )
    
    return train_data, val_data, test_data 