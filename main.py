import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model_1 import *
from data.generate_data import create_data, normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr  
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定数据量
n_company = 110
num_days = 1002  # 股价天数
num_features = 6  # 每个时间点有6个特征（如开盘价、收盘价等）
time_window = 20

# 定义超参数，后继需要进一步调整！！
beta = 5e-10
theta = 0.001
epochs = 10
batch_size = n_company
learning_rate = 0.0001
hidden_dim1 = 768
hidden_dim2 = 256
hidden_dim3=128
hidden_dim4=64
dropout = 0.5
# # 调整 beta 和 theta 值，增加其对依赖性和协同关系的惩罚
# beta = 1e-6
# theta = 0.01
# num_layers = 2

## 获取股票数据 
folder_path = '/root/firsttry/CMIN/CMIN-US/price/processed/'

## 获取文件夹中所有文件的文件名
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

## 初始化一个空的DataFrame用于存储合并后的数据
all_data = pd.DataFrame()

##迭代每个文件，读取文件并添加文件名列
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, sep='\t', header=None, names=['dt', 'open', 'high', 'low', 'close', 'adj close', 'volume'])  # 读取文件
    df['code'] = file_name[0:-4]  # 在DataFrame中添加一列，用于存储文件名
    all_data = pd.concat([all_data, df], ignore_index=True)  # 合并当前文件数据到总数据

# 保存合并后的DataFrame为pkl文件 (num_stock*num_features, index+file_name+num_features+)
output_path = 'my_data.pkl'
all_data.to_pickle(output_path)

#  初始化一个空的三维数组，用于存储股价数据
stock_data = np.zeros((n_company, num_days, num_features))

#  将数据加载到stock_data中
for i, stock_code in enumerate(all_data['code'].unique()):
    stock_data[i] = all_data[all_data['code'] == stock_code].iloc[:, 1:7].values

# # 数据归一化(单只股票)
# normalized_data = np.zeros_like(stock_data)
# for company_index in range(stock_data.shape[0]):  # 遍历所有公司
#     for features_index in range(stock_data.shape[2]):  # 遍历所有股票类型
#         stock_data_single_company = stock_data[company_index, :, features_index]
        
#         # Min-Max 归一化
#         min_val = np.min(stock_data_single_company)
#         max_val = np.max(stock_data_single_company)
        
#         normalized_data[company_index, :, features_index] = (stock_data_single_company - min_val) / (max_val - min_val)
# stock_data = normalized_data

# 数据归一化(整体进行归一)
normalized_data = np.zeros_like(stock_data)
for features_index in range(stock_data.shape[2]):  
    # 提取所有公司和所有天数的当前股票类型数据
    stock_type_data = stock_data[:, :, features_index]  
    
    # 计算全局最大值和最小值
    global_min = np.min(stock_type_data)
    global_max = np.max(stock_type_data)
    
    # 对该股票类型进行 Min-Max 归一化
    normalized_data[:, :, features_index] = (stock_type_data - global_min) / (global_max - global_min)
stock_data = normalized_data

# 获取结构邻接矩阵
correlation_path="/root/firsttry/data/topology_matrix.csv"
correlation_matrix = pd.read_csv(correlation_path)
sadj = correlation_matrix.iloc[0:n_company, 1:(n_company+1)].values
sadj[abs(sadj) < 0.3] = 0  # 将相关系数小于0.3的边删除
sadj = torch.tensor(sadj, dtype=torch.float32) 
sadj = sadj + sadj.T.mul(sadj.T > sadj) - sadj.mul(sadj.T > sadj)
sadj = normalize(sadj + torch.eye(sadj.size(0), dtype=torch.float32))
sadj = torch.from_numpy(sadj)

# 创建训练数据集和测试集
batch_data_list = create_data(stock_data, sadj, window=time_window)
train_data, temp_data = train_test_split(batch_data_list, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 定义损失函数
def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost

def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC

# 计算 IC, RankIC, ICIR, RankICIR
def calculate_metrics(list1, list2):
   
    values1 = [t.item() for t in list1]
    values2 = [t.item() for t in list2]
    
    # 如果预测值和真实值完全相同，直接返回 IC 和 RankIC 为 1（完美相关）
    if np.allclose(values1, values2):
        return 1.0, 1.0, 1.0, 1.0
    
    # 计算IC
    ic = np.corrcoef(values1, values2)[0, 1] if np.std(values1) != 0 else np.nan # 使用 NumPy 的 corrcoef 函数
   
    # 如果相关系数不可用（例如，返回 NaN），则返回 0
    if np.isnan(ic):
        ic = 0
    
    # 计算RankIC
    rank_ic, _ = spearmanr(values1, values2)  # 使用 SciPy 的 spearmanr 函数
    if np.isnan(rank_ic):
        rank_ic = 0

    # 计算 ICIR 和 RankICIR
    ic_std = np.std(np.array(values1) - np.array(values2))  # IC 标准差
    rank_ic_std = np.std(np.argsort(values1) - np.argsort(values2))  # RankIC 标准差

    icir = ic / ic_std if ic_std != 0 else np.nan  # 防止除以零
    rankicir = rank_ic / rank_ic_std if rank_ic_std != 0 else np.nan  # 防止除以零

    return ic, rank_ic, icir, rankicir

# 定义训练模型
def train_model(model, train_data, val_data, epochs, lr, beta, theta):
    print("Training...")
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
   
    # 创建一个列表来存储每个epoch的损失值
    loss_values = []
    val_loss_values = []  # 存储验证集的损失
    IC = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ic = []  # 每个epoch记录的IC值列表
        for i in train_data:
            x, edge_index, edge_attr, y, fadj, sadj= i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred, emb1, com1, com2, emb2, emb= model(x, sadj, fadj, edge_index)
            optimizer.zero_grad()
            y=y.view(-1,1)

            # 模型预测
            n = emb1.size(0)
            mse_loss = F.mse_loss(y_pred, y)
            # mae_loss = F.l1_loss(y_pred, y)
            # rmse_loss = torch.sqrt(mse_loss)
            # mape_loss = torch.mean(torch.abs((y - y_pred) / (y + 1e-8))) * 100
            loss_dep = (loss_dependence(emb1, com1, n) + loss_dependence(emb2, com2, n))/2
            loss_com = common_loss(com1,com2)
            loss = mse_loss + beta * loss_dep + theta * loss_com
            ic, rank_ic, icir, rankicir = calculate_metrics(y_pred, y)
            epoch_loss += loss.item()

            if not np.isnan(ic):
                epoch_ic.append(ic)  # 记录每个batch的IC

            loss.backward()
            optimizer.step()

        # 平均损失
        avg_loss = epoch_loss / len(train_data)
        loss_values.append(avg_loss)
        # 验证集评估
        val_loss = evaluate_model(model, val_data, beta, theta)
        val_loss_values.append(val_loss)
        
        # 计算每个epoch的IC平均值
        if epoch_ic:
            avg_ic = np.nanmean(epoch_ic)
        else:
            avg_ic = np.nan
        IC.append(avg_ic)  # 将每个epoch的IC值添加到IC列表

        scheduler.step(val_loss)  # 传入验证损失进行学习率调整

        # 输出每个epoch的训练损失和验证损失
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f},IC: {avg_ic:.4f}')
        
    return loss_values, val_loss_values, IC

# 模型验证
def evaluate_model(model, val_data, beta, theta):
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    with torch.no_grad():  # 在验证时不需要梯度计算
        for i in val_data:
            x, edge_index, edge_attr, y, fadj, sadj = i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred, emb1, com1, com2, emb2, emb = model(x, sadj, fadj, edge_index)
            
            y = y.view(-1, 1)
            
            # 模型预测
            n = emb1.size(0)
            mse_loss = F.mse_loss(y_pred, y)
            # mae_loss = F.l1_loss(y_pred, y)
            # rmse_loss = torch.sqrt(mse_loss)
            # mape_loss = torch.mean(torch.abs((y - y_pred) / (y + 1e-8))) * 100
            loss_dep = (loss_dependence(emb1, com1, n) + loss_dependence(emb2, com2, n))/2
            loss_com = common_loss(com1,com2)
            
            # 计算验证集损失
            loss = mse_loss + beta * loss_dep + theta * loss_com
            val_loss += loss.item()
    
    # 返回验证集平均损失
    avg_val_loss = val_loss / len(val_data)
    return avg_val_loss


# 模型测试
def test_model(model, test_data):
    print("Testing...")
    model.eval()
    mse_total_loss = 0.0
    mae_total_loss = 0.0
    rmse_total_loss = 0.0
    mape_total_loss = 0.0
    pred_values = []
    target_values = []
    with torch.no_grad():
        for i in test_data:
            x, edge_index, edge_attr, y, fadj, sadj= i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred, emb1, com1, com2, emb2, emb= model(x, sadj, fadj, edge_index)
            y = y.view(-1, 1)
            mse_loss = F.mse_loss(y_pred, y)
            mae_loss = F.l1_loss(y_pred, y)
            rmse_loss = torch.sqrt(mse_loss)
            non_zero_mask = y != 0  # 过滤掉真实值为零的样本
            mape_loss = torch.mean(torch.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / (y[non_zero_mask] + 1e-8))) * 100
            mse_total_loss += mse_loss.item()
            mae_total_loss += mae_loss.item()
            rmse_total_loss += rmse_loss.item()
            mape_total_loss += mape_loss.item()
            pred_values.append(y_pred)
            target_values.append(y)
    # 拼接所有预测值和目标值
    pred_values = torch.cat(pred_values, dim=0)
    target_values = torch.cat(target_values, dim=0)
    print(f"main_test mse_loss: {mse_total_loss / len(test_data)}")
    print(f"main_test mae_loss: {mae_total_loss / len(test_data)}")
    print(f"main_test rmse_loss: {rmse_total_loss / len(test_data)}")
    print(f"main_test mape_loss: {mape_total_loss / len(test_data)}")
    return pred_values, target_values

if __name__ == "__main__":
    print("main.py is being run directly")
    # 创建模型
    model = SFGCN(input_dim = time_window, hidden_dim1 = hidden_dim1, hidden_dim2 = hidden_dim2, hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4, dropout = dropout).to(device)

    # 训练模型
    loss_values, val_loss_values, IC = train_model(model, train_data, val_data, epochs=epochs, lr=learning_rate, beta=beta, theta=theta)

    # 绘制训练时的损失曲线
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss_values, marker='o', linestyle='-', color='b', label='train_loss')
    plt.plot(epochs_range, val_loss_values, marker='o', linestyle='-', color='g', label='val_loss')
    plt.plot(epochs_range, IC, marker='o', linestyle='-', color='r', label='ic')
    plt.title('Loss/IC Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss/IC', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('main.png')

    # 测试模型
    pred_values, target_values = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]
    
    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

    # 输出四个评价指标
    print(f'IC: {ic}')
    print(f'RankIC: {rank_ic}')
    print(f'ICIR: {icir}')
    print(f'RankICIR: {rankicir}')

    # 绘制实际值与预测值对比
    plt.figure()
    plt.style.use('ggplot')
    # 创建折线图
    plt.plot(target_values, label='real', color='blue')  # 实际值
    plt.plot(pred_values, label='forecast', color='red', linestyle='--')  # 预测值

    # 增强视觉效果
    plt.grid(True)
    plt.title('real vs forecast')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('main_testing_real_forecast.png')


