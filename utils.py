import torch
from torch import nn
from torch.nn import init

from datetime import datetime
import time

class DebugConfig():
    dataset='ETT'                  # 'MNIST', 'ETT', 'SELF'
    model_type='MyTransformer'     # 'MLP', 'LeNet', 'RNN', 'TorchTransformer', 'OriginalTransformer', 'MyTransformer'
    
    # 日志
    note = 'Debug.'

    # 维度
    d_model=16
    d_ffn=64
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
    epochs=2
    batch_size=64
    lr=0.0005
    dropout=0.1

    train_percent=0.99

class Config():
    dataset='ETT'                  # 'MNIST', 'ETT', 'SELF'
    model_type='MyTransformer'     # 'MLP', 'LeNet', 'RNN', 'TorchTransformer', 'OriginalTransformer', 'MyTransformer'
    
    # 日志
    note = ''

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

def write_log(config: Config, min_loss, peak_memory):
    with open('./log.txt', 'a', encoding='utf-8') as log:
        log.write('*' * 80 + '\n')
        # 日志
        log.write(config.note + '\n')

        formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write(formatted_time + '\n')
        
        # 模型损失
        log.write(f'model: {config.model_type}, min_loss: {min_loss:.4f}, peak_memory: {peak_memory:.3f}MB\n')

        # 时空Cost
        log.write(Recorder.get_avg_record())

        # 维度
        log.write(f'd_model: {config.d_model}, d_ffn: {config.d_ffn}, d_input: {config.d_input}, d_output: {config.d_output}\n')
        
        # 稀疏注意力超参数
        log.write(f'n_neighbor: {config.n_neighbor}, d_chunk: {config.d_chunk}\n')

        # 模型超参数
        log.write(f'nhead: {config.nhead}, num_encoder_layers: {config.num_encoder_layers}\n')

        # 训练超参数
        log.write(f'seq_len: {config.seq_len}, epochs: {config.epochs}, batch_size: {config.batch_size}, lr: {config.lr}, dropout: {config.dropout}\n\n')

    Recorder.clear()

class Recorder():
    phases = []
    # (str: float)
    start_time = {}
    # (str: [float, float, ...])
    delta_time = {}
    epoch_time = {}

    @staticmethod
    def start(phase):
        # 确保所有CUDA操作完成
        torch.cuda.synchronize()
        Recorder.start_time[phase] = time.time()

        if phase not in Recorder.phases:
            Recorder.phases.append(phase)
            Recorder.delta_time[phase] = 0
            Recorder.epoch_time[phase] = []

    @staticmethod
    def end(phase):
        torch.cuda.synchronize()
        delta_time = time.time() - Recorder.start_time[phase]
        Recorder.delta_time[phase] += delta_time

    @staticmethod
    def epoch_sum():
        for phase in Recorder.phases:
            Recorder.epoch_time[phase].append(Recorder.delta_time[phase])
            Recorder.delta_time[phase] = 0

    @staticmethod
    def get_epoch_time(phase):
        # s
        return Recorder.epoch_time[phase][-1]
    
    @staticmethod
    def get_avg_time(phase):
        # s
        return sum(Recorder.epoch_time[phase]) / len(Recorder.epoch_time[phase])

    @staticmethod
    def get_epoch_record():
        record = ('-' * 80 + '\n')
        record += f'{'Phase':<10}'
        for phase in Recorder.phases:
            record += f'{phase:<10}'

        record += f'\n{'Time /s':<10}'
        for phase in Recorder.phases:
            record += f'{Recorder.get_epoch_time(phase):<10.3f}'
        record += ('\n' + '-' * 80)
        return record

    @staticmethod
    def get_avg_record():
        record = ('-' * 80 + '\n')
        record += f'{'Phase':<10}'
        for phase in Recorder.phases:
            record += f'{phase:<10}'

        record += f'\n{'Time /s':<10}'
        for phase in Recorder.phases:
            record += f'{Recorder.get_avg_time(phase):<10.3f}'
        record += ('\n' + '-' * 80)
        return record

    @staticmethod
    def clear():
        Recorder.phases = []