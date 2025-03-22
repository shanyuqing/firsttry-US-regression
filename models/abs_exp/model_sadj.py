# 消融实验模型介绍：
# 仅使用结构相关性图（由大型语言模型生成）。
# 使用多层GCN处理结构相关性图，生成节点嵌入。
# 直接使用全连接层进行股价预测，不引入时间相关性图或注意力机制。
# 目的：验证结构相关性图对股价预测的贡献。

import torch
from .my_model_utils import BaseSFGCN, GCN, construct_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SFGCN(BaseSFGCN):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout):
        super(SFGCN, self).__init__(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout)
        # 只使用结构图的GCN
        self.SGCN1 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)

    def forward(self, x, sadj, fadj, edge_index):
        # 只通过结构图进行图卷积
        emb1 = self.SGCN1(x, sadj)
        emb = emb1
        
        # 使用基类的预测方法
        y_pred = self.predict(emb, edge_index)
        
        return y_pred, emb

