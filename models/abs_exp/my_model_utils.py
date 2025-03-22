import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.layers import GraphConvolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_graph(correlation_matrix):
    """
    基于相关系数矩阵构建图
    
    Args:
        correlation_matrix: 相关系数矩阵
        
    Returns:
        edge_index: 边的索引
        edge_attr: 边的权重
    """
    num_stocks = correlation_matrix.shape[0]
    edge_index = []
    edge_attr = []
    
    for i in range(num_stocks):
        for j in range(num_stocks):
            # 保留非零相关的边
            if correlation_matrix[i, j] != 0:  
                edge_index.append([i, j])
                edge_attr.append(correlation_matrix[i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

class GCN(nn.Module):
    """基础图卷积网络"""
    def __init__(self, input_dim, hidden_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class Attention(nn.Module):
    """注意力机制"""
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class GCNLayer(nn.Module):
    """PyTorch Geometric的GCN层封装"""
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index, edge_attr)

class BaseSFGCN(nn.Module):
    """SFGCN的基础类，包含共同的组件和方法"""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout):
        super(BaseSFGCN, self).__init__()
        self.dropout = dropout
        
        # 公共GCN
        self.CGCN = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)
        
        # 注意力相关参数
        self.a = nn.Parameter(torch.zeros(size=(hidden_dim2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(hidden_dim2)
        self.tanh = nn.Tanh()
        
        # 预测层
        self.gcn1 = GCNLayer(hidden_dim2, hidden_dim3)
        self.gcn2 = GCNLayer(hidden_dim3, hidden_dim4)
        self.fc = nn.Linear(hidden_dim4, 1)

    def predict(self, x, edge_index):
        """共同的预测逻辑"""
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        return self.fc(x) 