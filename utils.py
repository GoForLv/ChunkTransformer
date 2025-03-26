import torch
from torch import nn
from torch.nn import init

from datetime import datetime
import time

class DebugConfig():
    dataset='ETT'                  # 'MNIST', 'ETT', 'SELF'
    model_type='MyTransformer'     # 'MLP', 'LeNet', 'RNN', 'MyTransformer', 'TorchTransformer'
    
    # 日志
    note = '使用TorchTransformer，'

    # 维度
    d_model=4
    d_ffn=512
    d_input=7
    d_output=7

    # 稀疏注意力超参数
    n_neighbor=2
    d_chunk=2

    # 模型超参数
    nhead=2
    num_encoder_layers=6

    # 训练超参数
    seq_len=4
    epochs=50
    batch_size=4
    lr=0.0005
    dropout=0.1

    train_percent=0.99

class Config():
    dataset='ETT'                # 'MNIST', 'ETT', 'SELF'
    model_type='MyTransformer'     # 'MLP', 'LeNet', 'RNN', 'MyTransformer', 'TorchTransformer'
    
    # 日志
    note = '使用MyTransformer，传统多头注意力作为对照。'

    # 维度
    d_model=64
    d_ffn=256
    d_input=7
    d_output=7

    # 稀疏注意力超参数
    n_neighbor=4
    d_chunk=8

    # 模型超参数
    nhead=8
    num_encoder_layers=6

    # 训练超参数
    seq_len=256
    epochs=50
    batch_size=64
    lr=0.0005
    dropout=0.1

    train_percent=0.99

@torch.no_grad()
def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def write_log(config: Config, min_loss, note):
    with open('./log.txt', 'a', encoding='utf-8') as log:
        log.write('*' * 80 + '\n')
        # 日志
        log.write(note + '\n')

        formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write(formatted_time + '\n')
        log.write(f'model: {config.model_type}, min_loss: {min_loss:.4f}, Train Time: {Time.get()}, Attention Time: {Counter.get():.3f}\n')

        # 维度
        log.write(f'd_model: {config.d_model}, d_ffn: {config.d_ffn}, d_input: {config.d_input}, d_output: {config.d_output}\n')
        
        # 稀疏注意力超参数
        log.write(f'n_neighbor: {config.n_neighbor}\n')

        # 模型超参数
        log.write(f'nhead: {config.nhead}, num_encoder_layers: {config.num_encoder_layers}\n')

        # 训练超参数
        log.write(f'seq_len: {config.seq_len}, epochs: {config.epochs}, batch_size: {config.batch_size}, lr: {config.lr}, dropout: {config.dropout}\n\n')

class Time():
    start_time = None
    delta_time = 0

    @staticmethod
    def start():
        Time.start_time = time.time()
    
    @staticmethod
    def end():
        Time.delta_time += time.time() - Time.start_time

    @staticmethod
    def get():
        minute = int(Time.delta_time / 60)
        second = Time.delta_time % 60
        return f'{minute}m{second:.2f}s'
    
class Counter():
    start_time = None
    delta_time = 0

    @staticmethod
    def start():
        Counter.start_time = time.time()
    
    @staticmethod
    def end():
        Counter.delta_time += time.time() - Counter.start_time

    @staticmethod
    def get():
        # 毫秒
        return Counter.delta_time * 1000
