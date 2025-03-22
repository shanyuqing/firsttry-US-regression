import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.abs_exp.my_model_utils import BaseSFGCN, GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图卷积网络（SFGCN）
class SFGCN(BaseSFGCN):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout):
        super(SFGCN, self).__init__(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout)
        # 结构图和时间相关性图的GCN
        self.SGCN1 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)  # 处理结构图
        self.SGCN2 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)  # 处理特征图

    def forward(self, x, sadj, fadj, edge_index):
        # 通过结构图（sadj）和特征图（fadj）进行图卷积计算
        emb1 = self.SGCN1(x, sadj)  # Special_GCN1 -- 结构图
        com1 = self.CGCN(x, sadj)   # Common_GCN -- 结构图
        com2 = self.CGCN(x, fadj)   # Common_GCN -- 特征图
        emb2 = self.SGCN2(x, fadj)  # Special_GCN2 -- 特征图

        # 融合图卷积结果（结构图卷积和特征图卷积结果加权平均）
        Xcom = (com1 + com2) / 2

        # 堆叠所有图卷积结果
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        
        # 使用注意力机制进行加权
        emb, att = self.attention(emb)  # 计算加权的节点表示及其注意力权重
        
        # 使用基类的预测方法
        y_pred = self.predict(emb, edge_index)
        
        return y_pred, emb

