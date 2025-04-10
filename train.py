from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import Dataset
from utils import *
from model import MLP, LeNet, RNN, Transformer, TorchTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

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
        # self.prof = self._profiler()

        # 训练状态
        self.current_epoch = 0
        self.min_loss = 1e9
        self.peak_memory = 0

    def _model(self):
        if self.config.model_type == 'MLP':
            # model = MLP(in_dim=80, out_dim=1)
            model = MLP(in_dim=1 * 28 * 28, out_dim=10)
        elif self.config.model_type == 'LeNet':
            model = LeNet(num_classes=10)
        elif self.config.model_type == 'RNN':
            model = RNN(d_input=self.config.d_input, hidden_size=64, d_output=self.config.d_output)
        elif self.config.model_type == 'Torch':
            model = TorchTransformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout)
        elif self.config.model_type == 'Origin':
            model = Transformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout,
                                n_neighbor=self.config.n_neighbor,
                                d_chunk=self.config.d_chunk,
                                attn='Origin')
        elif self.config.model_type == 'Mask':
            model = Transformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout,
                                n_neighbor=self.config.n_neighbor,
                                d_chunk=self.config.d_chunk,
                                attn='Mask')
        elif self.config.model_type == 'Chunk':
            model = Transformer(d_model=self.config.d_model,
                                nhead=self.config.nhead,
                                d_ffn=self.config.d_ffn,
                                num_encoder_layers=self.config.num_encoder_layers,
                                d_input=self.config.d_input,
                                d_output=self.config.d_output,
                                dropout=self.config.dropout,
                                n_neighbor=self.config.n_neighbor,
                                d_chunk=self.config.d_chunk,
                                attn='Chunk')
        return model

    def _dataloader(self):
        if self.config.dataset == 'MNIST':
            self.train_dataset, self.test_dataset = Dataset.load_mnist_data()
        elif self.config.dataset == 'SELF':
            self.train_dataset, self.test_dataset = Dataset.load_self_data(seq_len=self.config.seq_len, train_percent=self.config.train_percent)
        elif self.config.dataset.startswith('ETT') :
            self.train_dataset, self.test_dataset = Dataset.load_ett_data(name=self.config.dataset, seq_len=self.config.seq_len, train_percent=self.config.train_percent)
        
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

    def _profiler(self):
        return profile(
                    record_shapes=True,   # 记录张量形状
                    profile_memory=True,  # 分析内存使用
                    with_stack=True,      # 记录调用栈
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],          # 分析 CPU 和 GPU
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'), # 保存到 ./logs, 以使用TensorBoard
                    # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2), # 采样策略
                )

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

            # self.prof.step()

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

    def train(self):
        for epoch in range(self.config.epochs):
            torch.cuda.reset_peak_memory_stats()

            self.current_epoch = epoch
            
            # 训练阶段
            Recorder.start('train')
            # self.prof.start()
            train_loss = self.train_epoch()
            # self.prof.stop()
            Recorder.pause('train')
            
            # 验证阶段
            Recorder.start('test')
            test_loss = self.test()
            Recorder.pause('test')

            if test_loss < self.min_loss:
                self.min_loss = test_loss

            # MB
            peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda")) / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, peak_memory)
            
            # 打印日志
            print(f'Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Min Loss: {self.min_loss:.4f}, Peak Memory: {peak_memory:.3f}MB.')

            print(Recorder.display_record('epoch'))
            # print(self.prof.key_averages().table(sort_by="cuda_time_total", row_limit=2))
        
        write_log(self.config, self.min_loss, self.peak_memory)

        # 可视化
        # self.visualize()

if __name__ == '__main__':
    import argparse
    import math

    parser = argparse.ArgumentParser(description='帮助文档')

    parser.add_argument('--data', help='ETTh1, ETTh2, ETTm1, ETTm2')
    parser.add_argument('--model', help='Torch, Origin, Chunk')
    parser.add_argument('--seq_len', type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--d_chunk', type=int, help='')

    args = parser.parse_args()

    config = Config()
    
    if args.data is not None:
        config.dataset = args.data

    if args.model is not None:
        config.model_type = args.model

    if args.seq_len is not None:
        config.seq_len = args.seq_len
    
    if args.epochs is not None:
        config.epochs = args.epochs

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.d_chunk is not None:
        config.d_chunk = args.d_chunk
        pass
    elif args.d_chunk == 0:
        x = int(math.log(args.seq_len, 2))
        factors = [i for i in range(1, args.seq_len + 1) if args.seq_len % i == 0]
        for factor in factors:
            if factor >= x:
                config.d_chunk = factor
                break
    else:
        config.d_chunk = args.d_chunk

    seq_lens = [256 * i for i in range(1, 17)]

    for seq_len in seq_lens:
        config.seq_len = seq_len
        if config.d_chunk == 0 and config.model_type == 'Chunk':
            x = int(math.log(config.seq_len, 2))
            factors = [i for i in range(1, config.seq_len + 1) if config.seq_len % i == 0]
            for factor in factors:
                if factor >= x:
                    config.d_chunk = factor
                    break

        config.display()
        trainer = Trainer(config)
        trainer.train()
