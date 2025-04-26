from copy import deepcopy
import matplotlib.pyplot as plt

from dataset import load_ett_data, load_mnist_data
from utils import *
from model import Transformer, TorchTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, config: Config, timer: Timer, logger: Logger):
        # 初始化配置
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化组件
        self.model = self._model().to(self.device)
        self.train_loader, self.val_loader, self.test_loader, self.scaler_mean, self.scaler_std = self._dataloader()
        self.optimizer = self._optimizer()
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.1, patience=3
        # )
        self.criterion = self._criterion()
        self.timer = timer
        self.logger = logger

        # 训练状态
        self.min_loss = 1e9
        self.best_model_state = None

        # 早停
        self.early_stop_patience = 12
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
            train_dataset, self.test_dataset = load_mnist_data()
            val_dataset = self.test_dataset
        elif self.config.dataset.startswith('ETT'):
            train_dataset, val_dataset, self.test_dataset, mean, std = load_ett_data(
                name=self.config.dataset,
                seq_len=self.config.seq_len,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio)

        return (
                DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
                DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False),
                DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False),
                mean.to(self.device), std.to(self.device)
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

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.timer.start("forward")
            outputs = self.model(inputs)
            self.timer.stop("forward")
            
            # 损失计算
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.timer.start("backward")
            loss.backward()
            # 梯度裁剪 避免梯度爆炸
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.timer.stop("backward")

            # 优化
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 批次损失均值 reduction='mean'
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                # print("outputs", outputs)
                # print("labels", labels)
                
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    # 反归一化
    def denormalize(self, data):
        return data * self.scaler_std + self.scaler_mean

    def test(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)

                origin_labels = self.denormalize(labels)
                origin_outputs = self.denormalize(outputs)

                loss = self.criterion(origin_outputs, origin_labels)
                
                test_loss += loss.item()

        return test_loss / len(self.test_loader)

    def load_best_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def visualize(self):
        '''
        multivariate多特征时序预测可视化
        '''
        inputs, labels = self.test_dataset.tensors
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        inputs = self.denormalize(inputs)
        outputs = self.denormalize(outputs)
        labels = self.denormalize(labels)

        count = self.config.d_output
        _, axes = plt.subplots(
            nrows = count,
            ncols=1,
            figsize=(16, 3 * count),
            sharex=False
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
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.grid(True)

        plt.legend()
        # plt.show()


    def train(self):
        for epoch in range(self.config.epochs):
            # 训练阶段
            self.timer.start('train')
            train_loss = self.train_epoch()
            self.timer.stop('train')
            
            # 验证阶段
            self.timer.start('validate')
            val_loss = self.validate()
            self.timer.stop('validate')

            # self.scheduler.step(val_loss)

            if val_loss < self.min_loss:
                self.min_loss = val_loss
                self.no_improve_epochs = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.no_improve_epochs += 1

            # 打印日志
            self.logger.write(f'Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Min Loss: {self.min_loss:.4f}, Peak Memory: {self.timer.peak_memory():.3f}MB.\n')
            self.logger.write(self.timer.display_record('epoch'))

            # 早停判断
            if self.no_improve_epochs >= self.early_stop_patience:
                self.logger.write(f'Early stopping at epoch {epoch+1}...\n')
                break

        self.load_best_model()
        # 可视化
        self.visualize()

        test_loss = self.test()
        self.logger.logger(self.min_loss, test_loss, self.best_model_state)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='帮助文档')

    parser.add_argument('--data', help='ETTh1, ETTh2, ETTm1, ETTm2')
    parser.add_argument('--model', help='Torch, Base, HBA')
    parser.add_argument('--seq_len', type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--d_block', type=int, help='')

    args = parser.parse_args()

    config = Config()
    timer = Timer()
    logger = Logger(config=config, timer=timer)
    
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
        # if args.model == 'HBA' and args.d_block == 0:
        #     config.d_block = int(math.log(args.seq_len, 2))
        trainer = Trainer(config, timer, logger)
        trainer.train()
    else:
        seq_lens = [128, 256, 512, 1024, 2048]
        for seq_len in seq_lens:
            config.seq_len = seq_len
            # if config.model_type == 'HBA' and config.d_block == 0:
            #     config.d_block = int(math.log(seq_len, 2))
            trainer = Trainer(config, timer, logger)
            trainer.train()