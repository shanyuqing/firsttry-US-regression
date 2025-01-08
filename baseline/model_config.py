from easydict import EasyDict

# 定义 Base_Config
Base_Config = EasyDict({
    'input_size': 20,
    'lr':0.0001,
    "hidden_size" :64,
    "output_size" : 1,# Output for each node (or aggregated output)
    "num_nodes" : 110  # Number of nodes (example size)
})

# 定义其他 config，例如 config_gat
Gat_Config = EasyDict({
    "num_heads" : 4,
    "epochs" :64,
    'Base': True  
})

# 待修改
Gru_Config = EasyDict({
    "num_layers" : 2,
    "epochs" :64,
    'Base': True  
})

# 待修改
Gru_gat_Config = EasyDict({
    "num_heads" : 4,
    "gru_layers" : 2,
    "epochs" :50,
    'Base': True  
})
# 待修改
Lstm_Config = EasyDict({
    "num_layers" : 2,
    "epochs" :128,
    'Base': True
})
# 待修改
Lstm_gat_Config = EasyDict({
    "num_heads" : 4,
    "lstm_layers" : 2,
    "epochs" :64,
    'Base': True  
})
Rnn_Config = EasyDict({
    "num_layers" : 2,
    "epochs" :32,
    'Base': True  
})
# 更新函数
def update_config_with_base(config, base_config):
    """
    如果 config 中 'Base' 键为 True，使用 base_config 更新 config。
    
    参数：
    - config: 待更新的 EasyDict 配置对象
    - base_config: Base 配置对象，用于更新
    
    返回：
    - 更新后的 config
    """
    if config.get('Base', False):  # 如果 'Base' 为 True
        config.update(base_config)  # 使用 base_config 更新 config
    return config



# 使用函数更新 config
Gat_Config = update_config_with_base(Gat_Config, Base_Config)
Gru_Config = update_config_with_base(Gru_Config, Base_Config)
Gru_gat_Config = update_config_with_base(Gru_gat_Config, Base_Config)
Lstm_Config = update_config_with_base(Lstm_Config, Base_Config)
Lstm_gat_Config = update_config_with_base(Lstm_gat_Config, Base_Config)
Rnn_Config = update_config_with_base(Rnn_Config, Base_Config)


if __name__ == '__main__':
    # 输出更新后的 config
    print(Gat_Config)
    print(Gru_Config)
    print(Gru_gat_Config)
    print(Lstm_Config)
    print(Lstm_gat_Config)
    print(Rnn_Config)

