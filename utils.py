import torch
from torch import nn

import math
import json
from datetime import datetime
import time
import os
import matplotlib.pyplot as plt

def softmax(scores: torch.Tensor, dim=-1):
    """Compute softmax with improved numerical stability.
    
    Args:
        scores: Input tensor containing unnormalized scores
        dim: Dimension along which softmax will be computed (default: -1)
    
    Returns:
        Tensor with same shape as input, with softmax applied along specified dimension
    """
    scores_max, _ = torch.max(scores, dim=dim, keepdim=True)
    scores_exp = torch.exp(scores - scores_max)
    attn = scores_exp / (scores_exp.sum(dim=dim, keepdim=True) + 1e-8)
    return attn

class PatchEmbedding(nn.Module):
    """将图像分割为patch并嵌入"""
    def __init__(self, embed_dim, img_size=32, patch_size=4, in_channels=1, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # use Conv2D to patch cutting and embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B, C, H, W] -> [B, E, N, N] -> [B, E, N*N] -> [B, N*N, E]
        # [B, E, H/P, W/P]
        x = self.proj(x)
        # [B, E, N*N]
        x = x.flatten(2)
        # [B, N*N, E]
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    """Inject positional information into input sequences using sine and cosine functions.
    
    Args:
        d_model: Dimension of the model embeddings
        dropout: Dropout probability
        max_len: Maximum length of input sequences (default: 4096)
    """
    def __init__(self, d_model, dropout, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 缓冲区 不更新参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert self.pe.size(1) >= x.size(1), 'max_len < seq_len!'
        
        # (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Config():
    def __init__(self):
        # 数据集配置
        self.dataset: str = 'MNIST'
        self.model_type: str = 'HBA'  # ['Torch', 'Base', 'HBA']
        
        # 模型维度配置
        self.d_model: int = 32
        self.d_ffn: int = 128
        self.d_input: int = 7
        self.d_output: int = 7
        
        # 训练配置
        self.seq_len: int = 512
        self.epochs: int = 300
        self.batch_size: int = 32
        self.lr: float = 0.0001
        self.dropout: float = 0.1

        # 数据集分割
        self.train_ratio: float = 0.7
        self.val_ratio: float = 0.2

        # 稀疏注意力超参数
        self.d_block=8

        # 模型超参数
        self.n_head=4
        self.num_encoder_layers=4

        # 验证参数合法性
        self._validate()
    
    def _validate(self):
        pass

    def display(self):
        print(f'data={self.dataset}, model={self.model_type}, epochs={self.epochs}, batch_size={self.batch_size}, seq_len={self.seq_len}, d_block={self.d_block}')

    def save(self, json_path):
        config_dict = {k: v for k, v in vars(self).items() if not k.startswith('_')}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(config_dict)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

class Timer():
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.phases = []
        # (str: float)
        self.start_time = {}
        self.delta_time = {}
        # (str: [float, float, ...])
        self.epoch_time = {}

        self._peak_memory = 0
        torch.cuda.reset_peak_memory_stats()

    def start(self, phase):
        # 确保所有CUDA操作完成
        torch.cuda.synchronize()
        self.start_time[phase] = time.perf_counter()

        if phase not in self.phases:
            self.phases.append(phase)
            self.delta_time[phase] = 0
            self.epoch_time[phase] = []

    def stop(self, phase):
        torch.cuda.synchronize()
        self.delta_time[phase] += time.perf_counter() - self.start_time[phase]
    
    def end_epoch(self):
        for phase in self.phases:
            self.epoch_time[phase].append(self.delta_time[phase])
            self.delta_time[phase] = 0

    def peak_memory(self):
        # MB
        peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda")) / (1024 ** 2)
        self._peak_memory = max(self._peak_memory, peak_memory)
        torch.cuda.reset_peak_memory_stats()
        return self._peak_memory

    def get_epoch_time(self, phase):
        # s
        return self.epoch_time[phase][-1]
    
    def get_avg_time(self, phase):
        # s
        if len(self.epoch_time[phase]) <= 5:
            return sum(self.epoch_time[phase]) / len(self.epoch_time[phase])
        return sum(self.epoch_time[phase][5:]) / len(self.epoch_time[phase][5:])

    def _display_record(self, get_time):
        record = ('-' * 80 + '\n')
        word = 'Phase'
        record += f'{word:<15}'
        for phase in self.phases:
            record += f'{phase:<10}'

        word = 'Epoch Time /s'
        record += f'\n{word:<15}'
        for phase in self.phases:
            record += f'{get_time(phase):<10.3f}'
        record += ('\n' + '-' * 80 + '\n')
        return record

    def display_record(self, type: str):
        '''
        param:
        type: 'epoch', 'average'
        '''
        if type == 'epoch':
            self.end_epoch()
            recorder = self._display_record(self.get_epoch_time)
        elif type == 'average':
            recorder = self._display_record(self.get_avg_time)
            self.reset()
        return recorder

class Logger():
    def __init__(self, config: Config, timer: Timer) -> None:
        self.today = datetime.now().strftime("%m-%d")
        self.counter = 0
        self.log_path = self._get_log_path()
        self.log_file = open(self.log_path, 'a', encoding='utf-8')

        self.config = config
        self.timer = timer

    def _get_log_path(self):
        while True:
            log_path = os.path.join(
                'log',
                'txtlog',
                f"{self.today}-{self.counter}.txt"
            )
            if not os.path.exists(log_path):
                break
            self.counter += 1
        return log_path

    def write(self, info: str):
        self.log_file.write(info)

    def logger(self, min_loss, test_loss, best_model_state):
        self.csv_logger(min_loss, test_loss)
        self.config.save(os.path.join(
            'log',
            'config',
            f"{self.today}-{self.counter}.json"
        ))
        self.log_file.write(f'Summary:, Min Loss: {min_loss:.4f}, Test Loss: {test_loss:.4f}, Peak Memory: {self.timer._peak_memory:.3f}MB\n')
        self.log_file.write(self.timer.display_record('average'))
        self.log_file.write('\n' * 2)
        plt.savefig(os.path.join(
            'log',
            'imglog',
            f"{self.today}-{self.counter}-{self.config.seq_len}.png"
        ))
        torch.save(best_model_state, os.path.join(
            'log',
            'model',
            f"{self.today}-{self.counter}.pth"
        ))

    def csv_logger(self, min_loss, test_loss):
        log_path = os.path.join(
                'log',
                'csvlog',
                f"{self.today}.csv"
            )
        is_empty = not os.path.exists(log_path) or os.stat(log_path).st_size == 0
        with open(log_path, 'a', encoding='utf-8') as log:
            if is_empty:
                log.write('log_path,model,seq_len,d_block,d_ffn,train,forward,backward,validate,min_loss,test_loss,peak_memory/MB\n')
            log.write(f"{self.today+'-'+str(self.counter)},{self.config.model_type},{self.config.seq_len},{self.config.d_block},{self.config.d_ffn},"
                    f"{self.timer.get_avg_time('train')},{self.timer.get_avg_time('forward')},"
                    f"{self.timer.get_avg_time('backward')},{self.timer.get_avg_time('validate')},"
                    f"{min_loss},{test_loss},{self.timer._peak_memory}\n")