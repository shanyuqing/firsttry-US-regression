import torch
from models.my_model import SFGCN
from data.data_util import get_train_val_test_data
from train_utils.train_eval import run_experiment

# 配置参数
class Config:
    # 数据相关
    n_company = 110
    num_days = 1002
    num_features = 6
    time_window = 18
    data_path = '/root/firsttry-main/firsttry_US/CMIN/processed'
    correlation_path = "/root/firsttry-main/firsttry_US/data/topology_matrix.csv"
    correlation_threshold = 0.7
    
    # 模型相关
    beta = 1.2164610467572498e-08
    theta = 1.3510247718823894e-10
    epochs = 38
    batch_size = n_company
    learning_rate = 5.0049754318915105e-05
    hidden_dim1 = 125
    hidden_dim2 = 260
    hidden_dim3 = 193
    hidden_dim4 = 248
    dropout = 0.8993095644168758

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Loading and processing data...")
    # 获取数据
    train_data, val_data, test_data = get_train_val_test_data(Config)

    # 创建模型
    model = SFGCN(
        input_dim=Config.time_window, 
        hidden_dim1=Config.hidden_dim1, 
        hidden_dim2=Config.hidden_dim2, 
        hidden_dim3=Config.hidden_dim3, 
        hidden_dim4=Config.hidden_dim4, 
        dropout=Config.dropout
    ).to(device)

    # 训练和评估模型
    results = run_experiment(model, train_data, val_data, test_data, Config, "SFGCN")

    # 打印结果
    print("\nFinal Results:")
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.6f}")



