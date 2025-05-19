import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv
import copy
import os
import math

# d_chunk = 8
class DataProcessor():
    def __init__(self, models) -> None:
        self.phases_dict = {
            # 'train': [],
            # 'forward': [],
            # 'backward': [],
            'validate': [],
            # 'min_loss': [],
            # 'test_loss': [],
            # 'peak_memory': [],
        }
        self.init_models = models
        self.data = {}
        self.models = []
        self.phases = list(self.phases_dict.keys())
        # for model in models:
        #     self.data[model] = copy.deepcopy(self.phases_dict)
    
    def add(self, log_path, model, seq_len, d_block, d_ffn, train, forward, backward, validate, min_loss, test_loss, peak_memory):
        if model not in self.init_models:
            return
        model_type = model + str(d_ffn)
        if model_type not in self.models:
            self.models.append(model_type)
            self.data[model_type] = copy.deepcopy(self.phases_dict)
        
        train = float(str(train)[len(str(d_ffn)):])
        # self.data[model_type]['train'].append(train)
        # self.data[model]['forward'].append(forward)
        # self.data[model]['backward'].append(backward)
        self.data[model_type]['validate'].append(validate)
        # self.data[model]['min_loss'].append(min_loss)
        # self.data[model_type]['test_loss'].append(test_loss)
        # self.data[model_type]['peak_memory'].append(peak_memory * 10)

    def get(self):
        for model in self.models:
            print(f'{model}:')
            for phases in self.phases:
                print(f'{phases:<15}{self.data[model][phases]}')
            print()

def polyfit(x: list, y: list, n: int) -> list:
    nx = np.array(x)
    ny = np.array(y)
    coefficients = np.polyfit(nx, ny, n)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = polynomial(x_fit)
    return x_fit, y_fit

def color(model):
    if model.startswith('Torch'):
        d_ffn = int(model[5:])
    elif model.startswith('Base'):
        d_ffn = int(model[4:])
    elif model.startswith('Linformer'):
        d_ffn = int(model[9:])
    elif model.startswith('HBA'):
        d_ffn = int(model[3:])

    if d_ffn == 64:
        return 'red'
    elif d_ffn == 128:
        return 'green'
    elif d_ffn == 256:
        return 'blue'
    else:
        print('d_ffn error!')

def visualize(processor):
    # seq_len = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    seq_len = [i+2 for i in range(6, 14)]
    num_phases = len(processor.phases)

    ncols = 1
    _, axes = plt.subplots(
        nrows=num_phases,
        ncols=ncols,
        # 列 行
        # figsize=(12, 3 * nrows),
        sharex=False
    )
    for idx, phase in enumerate(processor.phases):
        ax = axes
        for model in processor.models:
            nsamples = min(len(seq_len), len(processor.data[model][phase]))
            if phase == 'min_loss' or phase == 'test_loss':
                ax.plot(
                    seq_len[:nsamples],
                    processor.data[model][phase][:nsamples],
                    linestyle='-' if model.startswith('HBA') else '--',
                    label=model,
                    marker='o',
                )
            else:
                if model.startswith('HBA'):
                    ax.plot(
                        seq_len[:nsamples],
                        processor.data[model][phase][:nsamples],
                        linestyle='-',
                        color=color(model),
                        marker='o',
                        label=model,
                        lw=3,
                        markersize=10,
                    )
                else: 
                    # model.startswith('Base'):
                    ax.plot(
                        seq_len[:nsamples],
                        processor.data[model][phase][:nsamples],
                        linestyle='--',
                        color=color(model),
                        marker='o',
                        label=model,
                        lw=3,
                        markersize=10,
                    )

                # ax.plot(
                #     *polyfit(seq_len[:nsamples], processor.data[model][phase][:nsamples], 2)
                # )
        ax.grid()
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlabel(r'$log_2(L)$')
        
        if phase == 'min_loss':
            ax.set_ylabel('loss')
        elif phase == 'peak_memory':
            max_y = 20000
            ticks = [1024 * i for i in range(0, int(max_y/1024)+2, 4)]
            labels = [f"{1024 * i}" if i > 0 else "0" for i in range(0, int(max_y/1024)+2, 4)]

            plt.yticks(ticks, labels)
            ax.set_ylabel('Peak Memory(MB)', fontsize=14)
        else:
            ax.set_ylabel(r'$second / epoch$')

    custom_lines = [
        Line2D([0], [0], color='red', lw=0, marker='o'),
        Line2D([0], [0], color='green', lw=0, marker='o'),
        Line2D([0], [0], color='blue', lw=0, marker='o'),
        Line2D([0], [0], color='black', lw=3, linestyle='--'),
        Line2D([0], [0], color='black', lw=3, linestyle='-'),
    ]

    plt.legend(
        custom_lines,
        ['d_ffn=64', 'd_ffn=128', 'd_ffn=256', 'Transformer', 'HBAformer'],
        loc='upper left',
        fontsize=16,
    )

    plt.tick_params(axis='both', which='major', labelsize=14)  # 主刻度
    plt.tick_params(axis='both', which='minor', labelsize=12)  # 次刻度

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # models = ['Base', 'Torch', 'Linformer', 'HBA']
    models = ['Base', 'HBA']

    processor = DataProcessor(models)

    log_path = os.path.join(
        'log',
        'csvlog',
        'result.csv'
        # '05-11.csv'
    )
    with open(log_path, 'r', encoding='utf-8') as csvfile:
        # 创建csv阅读器
        csv_reader = list(csv.reader(csvfile))
        start_line = 165 - 1
        end_line = 325
        step = 2
        # 逐行读取
        for row_idx in range(start_line, end_line, step):
            row = csv_reader[row_idx]
            # print(row)
            data = [int(i) for i in row[2:5]]
            data.extend([float(i) for i in row[5:]])
            processor.add(*row[:2], *data)

    processor.get()
    visualize(processor)