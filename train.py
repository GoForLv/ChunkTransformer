from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import Dataset
from utils import *
from model import MLP, LeNet, RNN, Transformer, TorchTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, config):
        # 初始化配置
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化组件
        self.model = self._model().to(self.device)
        self.train_loader, self.test_loader = self._dataloader()
        self.optimizer = self._optimizer()
        self.criterion = self._criterion()

        # 训练状态
        self.current_epoch = 0
        self.min_loss = 1e9

    def _model(self):
        if self.config.model_type == 'MLP':
            # model = MLP(in_dim=80, out_dim=1)
            model = MLP(in_dim=1 * 28 * 28, out_dim=10)
        elif self.config.model_type == 'LeNet':
            model = LeNet(num_classes=10)
        elif self.config.model_type == 'RNN':
            model = RNN(d_input=self.config.d_input, hidden_size=64, d_output=self.config.d_output)
        elif self.config.model_type == 'MyTransformer':
            model = Transformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout,
                                n_neighbor=self.config.n_neighbor,
                                d_chunk=self.config.d_chunk,
                                is_original=False)
        elif self.config.model_type == 'OriginalTransformer':
            model = Transformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout,
                                n_neighbor=self.config.n_neighbor,
                                d_chunk=self.config.d_chunk,
                                is_original=True)
        elif self.config.model_type == 'TorchTransformer':
            model = TorchTransformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout)
        return model

    def _dataloader(self):
        if self.config.dataset == 'MNIST':
            self.train_dataset, self.test_dataset = Dataset.load_mnist_data()
        elif self.config.dataset == 'SELF':
            self.train_dataset, self.test_dataset = Dataset.load_self_data(seq_len=self.config.seq_len, train_percent=self.config.train_percent)
        elif self.config.dataset == 'ETT':
            self.train_dataset, self.test_dataset = Dataset.load_etth1_data(seq_len=self.config.seq_len, train_percent=self.config.train_percent)
        
        return (
                DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True),
                DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)
                )

    def _criterion(self):
        # return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _optimizer(self):
        # return torch.optim.SGD(params=self.model.parameters(),
        #                        lr=self.config.lr)
        return torch.optim.Adam(params=self.model.parameters(), lr=self.config.lr)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for inputs, labels in tqdm(self.train_loader, desc=f'train epoch {self.current_epoch+1}'):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def test(self):
        self.model.eval()
        test_loss = 0.0
        
        for inputs, labels in self.test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            test_loss += loss.item()

        return test_loss / len(self.test_loader)

    # def save_checkpoint(self, is_best=False):
    #     state = {
    #         "epoch": self.current_epoch,
    #         "state_dict": self.model.state_dict(),
    #         "optimizer": self.optimizer.state_dict(),
    #         "best_acc": self.best_acc
    #     }
        
    #     torch.save(state, os.path.join(self.config.save_dir, "train.txt"))
    #     if is_best:
    #         torch.save(state, os.path.join(self.config.save_dir, "best.txt"))

    def visualize(self):
        '''
        multivariate多特征时序预测可视化
        '''
        inputs, labels = self.test_dataset.tensors
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        count = self.config.d_output
        _, axes = plt.subplots(
            nrows = count,
            ncols=1,
            figsize=(16, 3 * count),
            sharex=True
        )
        for idx in range(count):
            ax = axes[idx]
            # 绘制原始序列
            ax.plot(
                labels.cpu().numpy()[:, idx],
                label='Truth'
            )
            
            # 绘制预测区间的真实值
            ax.plot(
                outputs.cpu().numpy()[:, idx],
                label=self.config.model_type
            )

        plt.legend()
        plt.show()
        pass

    def train(self):
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            Time.start()
            train_loss = self.train_epoch()
            Time.end()
            
            # 验证阶段
            test_loss = self.test()
            if test_loss < self.min_loss:
                self.min_loss = test_loss

            # 打印日志
            print(f'Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Min Loss: {self.min_loss:.4f}, Train Time: {Time.get()}.\n')
        
        write_log(self.config, self.min_loss, self.config.note)

        # 可视化
        self.visualize()

if __name__ == '__main__':
    config = DebugConfig()
    trainer = Trainer(config)
    trainer.train()

    # config = Config()
    # config.seq_len = 512
    # for i in range(3):
    #     trainer = Trainer(config)
    #     trainer.train()