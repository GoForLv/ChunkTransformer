import torch
from torch import nn
from torch.nn import init

from datetime import datetime
import time
import os

class Config():
    def __init__(self):
        # 数据集配置
        self.dataset: str = 'ETTh1'
        self.model_type: str = 'HBA'  # ['Torch', 'Base', 'HBA']
        
        # 模型维度配置
        self.d_model: int = 64
        self.d_ffn: int = 256
        self.d_input: int = 7
        self.d_output: int = 7
        
        # 训练配置
        self.seq_len: int = 256
        self.epochs: int = 30
        self.batch_size: int = 32
        self.lr: float = 0.01
        self.dropout: float = 0.1

        # 数据集分割
        self.train_ratio: float = 0.6
        self.val_ratio: float = 0.2

        # 稀疏注意力超参数
        self.d_block=8

        # 模型超参数
        self.nhead=8
        self.num_encoder_layers=6

        # 验证参数合法性
        self._validate()
    
    def _validate(self):
        pass

    def display(self):
        print(f'data={self.dataset}, model={self.model_type}, epochs={self.epochs}, batch_size={self.batch_size}, seq_len={self.seq_len}, d_block={self.d_block}')

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
        return sum(Recorder.epoch_time[phase][5:]) / len(Recorder.epoch_time[phase][5:])

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

def write_csv_log(config, min_loss, peak_memory):
    log_path = os.path.join('csvlog', 'hpc.csv')
    with open(log_path, 'a', encoding='utf-8') as log:
        # log.write('model,seq_len,d_chunk,train,forward,criterion,backward,optimizer,test,min_loss,peak_memory/MB\n')
        log.write(f"{config.model_type},{config.seq_len},{config.d_block},"
                f"{Recorder.get_avg_time('train')},{Recorder.get_avg_time('forward')},"
                f"{Recorder.get_avg_time('criterion')},{Recorder.get_avg_time('backward')},"
                f"{Recorder.get_avg_time('optimizer')},{Recorder.get_avg_time('test')},"
                f"{min_loss},{peak_memory}\n")

def write_log(config, min_loss, peak_memory):
    log_path = os.path.join('txtlog', datetime.now().strftime('%m-%d')+'.txt')
    write_csv_log(config, min_loss, peak_memory)

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
        log.write(f'd_block: {config.d_block}\n')

        # 模型超参数
        log.write(f'nhead: {config.nhead}, num_encoder_layers: {config.num_encoder_layers}\n')

        # 训练超参数
        log.write(f'seq_len: {config.seq_len}, epochs: {config.epochs}, batch_size: {config.batch_size}, lr: {config.lr}, dropout: {config.dropout}\n\n')

    Recorder.clear()

