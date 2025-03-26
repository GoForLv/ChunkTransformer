from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import torch

import pandas as pd
import os

class Dataset():
    @staticmethod
    def load_etth1_data(seq_len, train_percent):
        data_path = os.path.join('data', 'ETT-small', 'ETTh1.csv')
        # Index(['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
        # shape: (17420, 8)
        df = pd.read_csv(data_path)
        data = torch.tensor(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values, dtype=torch.float32)

        num_samples = data.shape[0]
        seq_data = []
        for i in range(num_samples - seq_len):
            seq_data.append(data[i: i + seq_len + 1].tolist())
        # 17390 * 31 * 7
        seq_data = torch.tensor(seq_data)
        num_seq_data, _ = len(seq_data), len(seq_data[0])
        # 17216
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

    @staticmethod
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
    
    @staticmethod
    def load_self_data(seq_len, train_percent):
        num_samples = 5000 # -80
        X = torch.linspace(-1000, 1000, num_samples)
        T = 20
        y = torch.sin(2 * torch.pi / T * X) + torch.exp(X / 1000)

        seq_data = []
        for i in range(num_samples - seq_len):
            seq_data.append(y[i: i + seq_len + 1].tolist())
        seq_data = torch.tensor(seq_data)

        num_seq_data, _ = len(seq_data), len(seq_data[0])
        num_train = int(num_seq_data * train_percent)

        # 1344 * 81 * 1
        seq_train = seq_data[: num_train, :].view(num_train, -1, 1)
        # 576 * 81 * 1
        seq_test = seq_data[num_train: , :].view(num_seq_data - num_train, -1, 1)

        seq_train_X = seq_train[:, :-1, :]
        seq_train_y = seq_train[:, -1, :]
        seq_test_X = seq_test[:, :-1, :]
        seq_test_y = seq_test[:, -1, :]

        train_dataset = TensorDataset(seq_train_X, seq_train_y)
        test_dataset = TensorDataset(seq_test_X, seq_test_y)

        return train_dataset, test_dataset