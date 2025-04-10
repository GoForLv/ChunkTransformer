import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

# d_chunk = 8
class DataProcessor():
    def __init__(self, *models) -> None:
        self.phases_dict = {
            'train': [],
            'forward': [],
            'criterion': [],
            'backward': [],
            'optimizer': [],
            'test': [],
            'min_loss': [],
            'peak_memory': [],
        }
        self.data = {}
        self.models = models
        self.phases = self.phases_dict.keys()
        for model in models:
            self.data[model] = copy.deepcopy(self.phases_dict)
    
    def add(self, model, seq_len, d_chunk, train, forward, criterion, backward, optimizer, test, min_loss, peak_memory):
        self.data[model]['train'].append(train)
        self.data[model]['forward'].append(forward)
        self.data[model]['criterion'].append(criterion)
        self.data[model]['backward'].append(backward)
        self.data[model]['optimizer'].append(optimizer)
        self.data[model]['test'].append(test)
        self.data[model]['min_loss'].append(min_loss)
        self.data[model]['peak_memory'].append(peak_memory)

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

def visualize(processor):
    seq_len = [128, 256, 512, 1024]
    num_phases = len(processor.phases)

    ncols = 2
    nrows = num_phases // ncols
    _, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        # 列 行
        figsize=(12, 2 * nrows),
        sharex=True
    )
    for idx, phase in enumerate(processor.phases):
        ax = axes[idx//ncols][idx%ncols]
        for model in processor.models:
            ax.scatter(
                seq_len,
                processor.data[model][phase],
                label=model,
                marker='o',
            )
            ax.plot(
                *polyfit(seq_len, processor.data[model][phase], 2)
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
    processor = DataProcessor('TorchTransformer', 'MultiHeadTransformer', 'ChunkTransformer')

    with open('csvlog\log.csv', 'r', encoding='utf-8') as csvfile:
        # 创建csv阅读器
        csv_reader = list(csv.reader(csvfile))
        start_line = 6 - 1
        end_line = 50
        # 逐行读取
        for row_idx in range(start_line, end_line, 4):
            row = csv_reader[row_idx]
            data = [float(i) for i in row[3:]]
            processor.add(*row[0:3], *data)

    processor.get()
    visualize(processor)