from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import torch

import pandas as pd
import os

def load_mnist_data():
    '''
    参数:
    batch_size: 批量大小

    返回:
    train_loader, test_loader
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 60000 * tuple(tensor(1, 28, 28) float, int)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 10000
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def load_ett_data(name, seq_len, train_ratio=0.7, val_ratio=0.2):
    """
    加载ETT数据集并进行预处理
    参数:
        name: 数据集名称 (如 'ETTh1')
        seq_len: 序列长度
        train_ratio: 训练集比例 (默认0.7)
        val_ratio: 验证集比例 (默认0.2)
        测试集比例 = 1 - train_ratio - val_ratio
    返回:
        train_dataset: 训练集 TensorDataset
        val_dataset: 验证集 TensorDataset
        test_dataset: 测试集 TensorDataset
        scaler_mean: 训练集的均值 (用于反归一化)
        scaler_std: 训练集的标准差 (用于反归一化)
    """
    data_path = os.path.join('data', 'ETT-small', name+'.csv')
    df = pd.read_csv(data_path)
    
    # 提取特征数据并转换为tensor
    data = torch.tensor(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values, 
                       dtype=torch.float32)
    
    # 创建序列数据
    num_samples = data.shape[0]
    seq_data = []
    for i in range(num_samples - seq_len):
        seq_data.append(data[i: i + seq_len + 1])
    seq_data = torch.stack(seq_data)  # (num_seq, seq_len+1, num_features)
    
    # 分割数据集
    num_seq = seq_data.shape[0]
    num_train = int(num_seq * train_ratio)
    num_val = int(num_seq * val_ratio)
    
    # 训练集、验证集、测试集
    train_data = seq_data[:num_train]
    val_data = seq_data[num_train:num_train+num_val]
    test_data = seq_data[num_train+num_val:]
    
    # 计算归一化参数(仅使用训练集)
    scaler_mean = train_data.mean(dim=(0,1))  # (num_features,)
    scaler_std = train_data.std(dim=(0,1))    # (num_features,)
    
    # 归一化
    def normalize(_data):
        return (_data - scaler_mean) / (scaler_std + 1e-8)
    
    train_data = normalize(train_data)
    val_data = normalize(val_data)
    test_data = normalize(test_data)
    
    # 分割输入输出
    def create_dataset(data):
        X = data[:, :-1, :]  # (num_samples, seq_len, num_features)
        y = data[:, -1, :]   # (num_samples, num_features)
        return TensorDataset(X, y)
    
    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    test_dataset = create_dataset(test_data)
    
    return train_dataset, val_dataset, test_dataset, scaler_mean, scaler_std

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset, mean, std = load_ett_data('ETTh1', seq_len=2)
