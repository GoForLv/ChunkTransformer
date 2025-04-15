import matplotlib.pyplot as plt

if __name__ == '__main__':
    seq_len = [128, 256, 512, 1024]
    d_block = [8, 16, 32, 64]
    for m in d_block:
        print(f'分块大小 m={m}, L={seq_len}')
        accelerate = []
        for L in seq_len:
            accelerate.append(L / (m + L / m / m))
        print(accelerate)
        plt.plot(seq_len, accelerate)
        plt.show()