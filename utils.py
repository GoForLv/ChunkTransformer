import torch
from torch import nn
from torch.nn import init

from datetime import datetime
import time
import os

class DebugConfig():
    dataset='ETTh1'                     # 'MNIST', 'ETT', 'SELF'
    model_type='ChunkTransformer'

    # 维度
    d_model=16
    d_ffn=32
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
    epochs=12
    batch_size=128
    lr=0.0005
    dropout=0.1

    train_percent=0.80

class Config():
    dataset='ETTm1'                  # 'MNIST', 'ETT', 'SELF'
    model_type='ChunkTransformer'

    # 维度
    d_model=64
    d_ffn=256
    d_input=7
    d_output=7

    # 稀疏注意力超参数
    n_neighbor=64
    d_chunk=8

    # 模型超参数
    nhead=8
    num_encoder_layers=6

    # 训练超参数
    seq_len=256
    epochs=50
    batch_size=128
    lr=0.0005
    dropout=0.1

    train_percent=0.80

class Recorder():
    phases = []
    # (str: float)
    start_time = {}
    delta_time = {}
    # (str: [float, float, ...])
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
    def pause(phase):
        torch.cuda.synchronize()
        delta_time = time.time() - Recorder.start_time[phase]
        Recorder.delta_time[phase] += delta_time
    
    @staticmethod
    def end():
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
    def _display_record(get_time):
        record = ('-' * 80 + '\n')
        word = 'Phase'
        record += f'{word:<15}'
        for phase in Recorder.phases:
            record += f'{phase:<10}'

        word = 'Epoch Time /s'
        record += f'\n{word:<15}'
        for phase in Recorder.phases:
            record += f'{get_time(phase):<10.3f}'
        record += ('\n' + '-' * 80 + '\n')
        return record

    @staticmethod
    def display_record(type: str):
        '''
        param:
        type: 'epoch', 'average'
        '''
        if type == 'epoch':
            Recorder.end()
            recorder = Recorder._display_record(Recorder.get_epoch_time)
        elif type == 'average':
            recorder = Recorder._display_record(Recorder.get_avg_time)
        return recorder

    @staticmethod
    def clear():
        Recorder.phases = []
        Recorder.start_time = {}
        Recorder.delta_time = {}
        Recorder.epoch_time = {}

@torch.no_grad()
def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def write_csv_log(config, min_loss, peak_memory):
    log_path = os.path.join('csvlog', 'log.csv')
    with open(log_path, 'a', encoding='utf-8') as log:
        # log.write('model,seq_len,d_chunk,train,forward,criterion,backward,optimizer,test,min_loss,peak_memory/MB\n')
        log.write(f"{config.model_type},{config.seq_len},{config.d_chunk},\
                {Recorder.get_avg_time('train')},{Recorder.get_avg_time('forward')},\
                {Recorder.get_avg_time('criterion')},{Recorder.get_avg_time('backward')},\
                {Recorder.get_avg_time('optimizer')},{Recorder.get_avg_time('test')},\
                {min_loss},{peak_memory}\n")

def write_log(config, min_loss, peak_memory):
    if type(config) == Config:
        log_path = os.path.join('txtlog', datetime.now().strftime('%m-%d')+'.txt')
        write_csv_log(config, min_loss, peak_memory)
    elif type(config) == DebugConfig:
        log_path = os.path.join('txtlog', datetime.now().strftime('%m-%d')+'-debug.txt')

    with open(log_path, 'a', encoding='utf-8') as log:
        log.write('*' * 80 + '\n')

        formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write(formatted_time + '\n')
        
        # 模型损失
        log.write(f'model: {config.model_type}, min_loss: {min_loss:.4f}, peak_memory: {peak_memory:.3f}MB\n')

        # 时空Cost
        log.write(Recorder.display_record('average'))

        # 维度
        log.write(f'd_model: {config.d_model}, d_ffn: {config.d_ffn}, d_input: {config.d_input}, d_output: {config.d_output}\n')
        
        # 稀疏注意力超参数
        log.write(f'n_neighbor: {config.n_neighbor}, d_chunk: {config.d_chunk}\n')

        # 模型超参数
        log.write(f'nhead: {config.nhead}, num_encoder_layers: {config.num_encoder_layers}\n')

        # 训练超参数
        log.write(f'seq_len: {config.seq_len}, epochs: {config.epochs}, batch_size: {config.batch_size}, lr: {config.lr}, dropout: {config.dropout}\n\n')

    Recorder.clear()

