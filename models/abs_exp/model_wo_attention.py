# 消融实验模型介绍：
# 同时使用结构相关性图和时间相关性图。
# 分别使用多层GCN处理两种图，生成结构嵌入和时间嵌入。
# 直接拼接两种嵌入（而非使用注意力机制加权融合）。
# 使用全连接层进行股价预测。
# 目的：验证同时使用两种图结构的有效性，以及注意力机制的必要性。

import torch
from .my_model_utils import BaseSFGCN, GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SFGCN(BaseSFGCN):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout):
        super(SFGCN, self).__init__(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout)
        # 结构图和时间相关性图的GCN
        self.SGCN1 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)
        self.SGCN2 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)

    def forward(self, x, sadj, fadj, edge_index):
        # 通过两种图进行图卷积
        emb1 = self.SGCN1(x, sadj)  # 结构图
        com1 = self.CGCN(x, sadj)   # 结构图的公共特征
        com2 = self.CGCN(x, fadj)   # 时间图的公共特征
        emb2 = self.SGCN2(x, fadj)  # 时间图

        # 简单平均融合，不使用注意力机制
        Xcom = (com1 + com2) / 2
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb = emb.sum(1) / 3
        
        # 使用基类的预测方法
        y_pred = self.predict(emb, edge_index)
        
        return y_pred, emb1, com1, com2, emb2, emb

