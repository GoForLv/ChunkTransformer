from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import load_ett_data, load_mnist_data
from utils import *
from model import Transformer, TorchTransformer

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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3
        )
        self.criterion = self._criterion()

        # 训练状态
        self.current_epoch = 0
        self.min_loss = 1e9
        self.peak_memory = 0

        # 早停
        self.early_stop_patience = 6
        self.no_improve_epochs = 0

    def _model(self):
        if self.config.model_type == 'Torch':
            model = TorchTransformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout)
        elif self.config.model_type == 'Base':
            model = Transformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout,
                                d_block=self.config.d_block,
                                attn='Base')
        elif self.config.model_type == 'HBA':
            model = Transformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout,
                                d_block=self.config.d_block,
                                attn='HBA')
        return model

    def _dataloader(self):
        if self.config.dataset == 'MNIST':
            self.train_dataset, self.test_dataset = load_mnist_data()
        elif self.config.dataset.startswith('ETT') :
            self.train_dataset, self.test_dataset = load_ett_data(name=self.config.dataset, seq_len=self.config.seq_len, train_percent=self.config.train_percent)
        
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
        return torch.optim.Adam(params=self.model.parameters(),
                                lr=self.config.lr,
                                weight_decay=1e-4)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for inputs, labels in tqdm(self.train_loader, desc=f'train epoch {self.current_epoch+1}'):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            Recorder.start("forward")
            outputs = self.model(inputs)
            Recorder.pause("forward")
            
            # 损失计算
            Recorder.start("criterion")
            loss = self.criterion(outputs, labels)
            Recorder.pause("criterion")
            
            # 反向传播
            Recorder.start("backward")
            loss.backward()
            Recorder.pause("backward")

            # 优化
            Recorder.start("optimizer")
            self.optimizer.step()
            self.optimizer.zero_grad()
            Recorder.pause("optimizer")
            
            # 批次损失均值 reduction='mean'
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def test(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()

        self.scheduler.step(test_loss)
        return test_loss / len(self.test_loader)

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

    def train(self):
        for epoch in range(self.config.epochs):
            torch.cuda.reset_peak_memory_stats()

            self.current_epoch = epoch
            
            # 训练阶段
            Recorder.start('train')
            train_loss = self.train_epoch()
            Recorder.pause('train')
            
            # 验证阶段
            Recorder.start('test')
            test_loss = self.test()
            Recorder.pause('test')

            if test_loss < self.min_loss:
                self.min_loss = test_loss
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            # MB
            peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda")) / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, peak_memory)
            
            # 打印日志
            print(f'Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Min Loss: {self.min_loss:.4f}, Peak Memory: {peak_memory:.3f}MB.')
            print(Recorder.display_record('epoch'))

            if self.no_improve_epochs >= self.early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}...')
                break

        write_log(self.config, self.min_loss, self.peak_memory)

        # 可视化
        # self.visualize()

if __name__ == '__main__':
    import argparse
    import math

    parser = argparse.ArgumentParser(description='帮助文档')

    parser.add_argument('--data', help='ETTh1, ETTh2, ETTm1, ETTm2')
    parser.add_argument('--model', help='Torch, Base, HBA')
    parser.add_argument('--seq_len', type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--d_block', type=int, help='')

    args = parser.parse_args()

    config = Config()
    
    if args.data is not None:
        config.dataset = args.data

    if args.model is not None:
        config.model_type = args.model

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.d_block is not None:
        config.d_block = args.d_block

    if args.seq_len is not None:
        config.seq_len = args.seq_len
        if args.model == 'HBA' and args.d_block == 0:
            d_block = int(math.log(args.seq_len, 2))
            if d_block % 2 != 0:
                d_block += 1
            config.d_block = d_block

        config.display()
        trainer = Trainer(config)
        trainer.train()
    else:
        seq_lens = [128, 256, 512, 1024]
        for seq_len in seq_lens:
            config.seq_len = seq_len
            if args.model == 'HBA' and args.d_block == 0:
                d_block = int(math.log(seq_len, 2))
                if d_block % 2 != 0:
                    d_block += 1
                config.d_block = d_block

            config.display()
            trainer = Trainer(config)
            trainer.train()