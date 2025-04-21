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

def load_ett_data(name, seq_len, train_percent=0.7):
    """
    加载ETT数据集并进行预处理
    参数:
        name: 数据集名称 (如 'ETTh1')
        seq_len: 序列长度
        train_percent: 训练集比例 (默认0.7)
        val_percent: 验证集比例 (默认0.2)
        测试集比例 = 1 - train_percent - val_percent
    返回:
        train_dataset, val_dataset, test_dataset: 归一化后的TensorDataset
        scaler_mean: 训练集的均值 (用于反归一化)
        scaler_std: 训练集的标准差 (用于反归一化)
    """
    data_path = os.path.join('data', 'ETT-small', name+'.csv')
    # Index(['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
    # shape: (num_samples, 8)
    df = pd.read_csv(data_path)
    data = torch.tensor(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values, dtype=torch.float64)

    num_samples = data.shape[0]
    seq_data = []
    for i in range(num_samples - seq_len):
        seq_data.append(data[i: i + seq_len + 1].tolist())
    # (num_samples - seq_len) * (seq_len + 1) * 7
    seq_data = torch.tensor(seq_data)
    num_seq_data, _ = len(seq_data), len(seq_data[0])

    num_train = int(num_seq_data * train_percent)

    seq_train = seq_data[: num_train]
    seq_test = seq_data[num_train:]

    seq_train_X = seq_train[:, :-1, :]
    seq_train_y = seq_train[:, -1, :]
    seq_test_X = seq_test[:, :-1, :]
    seq_test_y = seq_test[:, -1, :]

    train_dataset = TensorDataset(seq_train_X, seq_train_y)
    test_dataset = TensorDataset(seq_test_X, seq_test_y)

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = load_ett_data('ETTh1', seq_len=2, train_percent=0.7)
