import numpy as np
import matplotlib.pyplot as plt
import csv
import copy
import math

# d_chunk = 8
class DataProcessor():
    def __init__(self, models) -> None:
        self.phases_dict = {
            'train': [],
            'forward': [],
            'backward': [],
            'validate': [],
            'min_loss': [],
            'peak_memory': [],
        }
        self.data = {}
        self.models = models
        self.phases = list(self.phases_dict.keys())
        for model in models:
            self.data[model] = copy.deepcopy(self.phases_dict)
    
    def add(self, log_path, model, seq_len, d_chunk, train, forward, backward, validate, min_loss, test_loss, peak_memory):
        if model not in self.models:
            return
        self.data[model]['train'].append(train)
        self.data[model]['forward'].append(forward)
        self.data[model]['backward'].append(backward)
        self.data[model]['validate'].append(validate)
        self.data[model]['min_loss'].append(min_loss)
        self.data[model]['peak_memory'].append(peak_memory)

    def get(self):
        for model in self.models:
            print(f'{model}:')
            for phases in self.phases:
                print(f'{phases:<15}{self.data[model][phases]}')
            print()
    
    def process(self, base, optim, compare, m, phase):
        '''Base, Optim, Optim/Base'''
        print('#' * 50)
        print(phase)
        n1 = len(self.data[base][phase])
        n2 = len(self.data[optim][phase])
        n = min(n1, n2)
        seq_len = [128, 256, 512, 1024, 2048, 3072, 4096][:n]
        if phase == 'min_loss' or phase == 'fake_loss':
            for i in range(n):
                self.data[compare][phase].append((self.data[optim][phase][i] - self.data[base][phase][i]) / self.data[base][phase][i] * 100)
            print(base, self.data[base][phase])
            print(optim, self.data[optim][phase])
            print(compare, 'relative loss:', self.data[compare][phase])
            plt.plot(seq_len[:n], self.data[compare][phase], marker='o', label=compare)
            plt.xlabel('seq_len')
            plt.ylabel('%')
            plt.legend(loc='upper left', fontsize=8)
            plt.show()
            return

        for i in range(n):
            self.data[compare][phase].append(self.data[base][phase][i] / self.data[optim][phase][i])

        print(f'{base}: {self.data[base][phase][:n]}')
        print(f'{optim}: {self.data[optim][phase][:n]}')
        print(f'{compare}: {self.data[compare][phase][:n]}')

        base_time = [L * L for L in seq_len]

        if m == 0:
            m = [8, 8, 16, 16, 16]
            optim_time = [seq_len[i] * m[i] for i in range(n)]
        else:
            optim_time = [L * m for L in seq_len]

        compare_time = [base_time[i] / optim_time[i] for i in range(n)]

        _, axes = plt.subplots(
            nrows=1,
            ncols=3,
            # 列 行
            figsize=(12, 4 * 1),
            sharex=False
        )
        axes[0].plot(self.data[base][phase][:n], base_time, marker='o')
        axes[0].set_title(base)

        axes[1].plot(self.data[optim][phase][:n], optim_time, marker='o')
        axes[1].set_title(optim)

        axes[2].plot(self.data[compare][phase][:n], compare_time, marker='o')
        axes[2].set_title(compare)

        plt.show()

def polyfit(x: list, y: list, n: int) -> list:
    nx = np.array(x)
    ny = np.array(y)
    coefficients = np.polyfit(nx, ny, n)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = polynomial(x_fit)
    return x_fit, y_fit

def visualize(processor):
    # seq_len = [128, 256, 512, 1024]
    seq_len = [1, 2, 4, 8]
    num_phases = len(processor.phases)

    ncols = 2
    nrows = num_phases // 2
    _, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        # 列 行
        # figsize=(12, 3 * nrows),
        sharex=False
    )
    for idx, phase in enumerate(processor.phases):
        ax = axes[idx//ncols][idx%ncols]
        for model in processor.models:
            nsamples = min(len(seq_len), len(processor.data[model][phase]))
            if phase == 'min_loss':
                ax.plot(
                    seq_len[:nsamples],
                    processor.data[model][phase][:nsamples],
                    label=model,
                    marker='o',
                )
            else:
                ax.scatter(
                    seq_len[:nsamples],
                    processor.data[model][phase][:nsamples],
                    label=model,
                    marker='o',
                )
                ax.plot(
                    *polyfit(seq_len[:nsamples], processor.data[model][phase][:nsamples], 2)
                )
        ax.grid()
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlabel('seq_len')
        if phase == 'min_loss':
            ax.set_ylabel('loss')
        elif phase == 'peak_memory':
            ax.set_ylabel('MB')
        else:
            ax.set_ylabel('sec /epoch')
        ax.set_title(phase)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 旧数据
    # models = ['Base', 'Torch', 'HBA_16']

    # 新数据 所有模型对比
    models = ['Base', 'Torch', 'HBA', 'FullHBA']
    # models = ['Base', 'Torch', 'HBA', 'HBA_8', 'HBA_16', 'HBA_32', 'HBA_64']

    # 验证加速比
    # models = ['Base', 'HBA_8', 'HBA_8/Base', 'Torch', 'Torch/Base']
    processor = DataProcessor(models)

    with open('log\csvlog\\05-05.csv', 'r', encoding='utf-8') as csvfile:
        # 创建csv阅读器
        csv_reader = list(csv.reader(csvfile))
        if False:
            # old
            start_line = 5 - 1
            end_line = 45
            # 逐行读取
            for row_idx in range(start_line, end_line, 4):
                row = csv_reader[row_idx]
                # print(row)
                data = [float(i) for i in row[3:]]
                processor.add(*row[0:3], *data)
        else:
            # new
            start_line = 3 - 1
            end_line = 33
            step = 2
            # 逐行读取
            for row_idx in range(start_line, end_line, step):
                row = csv_reader[row_idx]
                # print(row)
                data = [float(i) for i in row[4:]]
                processor.add(*row[:4], *data)

    # processor.process('Origin', 'Chunk0', 'Chunk0/Origin')
    # for phase in ['min_loss']:
    #     processor.process('Base', 'HBA_8', 'HBA_8/Base', m=8, phase=phase)
    #     processor.process('Base', 'Torch', 'Torch/Base', m=8, phase=phase)

    # for phase in ['train', 'forward', 'backward', 'test', 'peak_memory', 'min_loss']:
    #     processor.process('Base', 'Torch', 'Torch/Base', m=8, phase=phase)
    
    # processor.process('Origin', 'Chunk16', 'Chunk16/Origin')
    # processor.process('Origin', 'Chunk32', 'Chunk32/Origin')
    # processor.process('Origin', 'Chunk64', 'Chunk64/Origin')
    processor.get()
    visualize(processor)