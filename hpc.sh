#!/bin/bash
#SBATCH -J pytorch_test
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -o main.out
#SBATCH -e main.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

ulimit -u unlimited
export PATH="/share/apps/anaconda3/bin:$PATH"
source activate pytorch-env

date

echo "Train..."

python train.py --data=ETTh1 --epochs=300 --batch_size=64 --model=Base
# python train.py --data=ETTh1 --epochs=300 --batch_size=64 --model=Torch
# python train.py --data=ETTh1 --epochs=300 --batch_size=64 --model=Linformer
# python train.py --data=ETTh1 --epochs=300 --batch_size=64 --model=LocalHBA --d_block=8
python train.py --data=ETTh1 --epochs=300 --batch_size=64 --model=HBA --d_block=8

python train.py --data=ETTh1 --epochs=300 --batch_size=32 --model=Base
python train.py --data=ETTh1 --epochs=300 --batch_size=32 --model=Torch
python train.py --data=ETTh1 --epochs=300 --batch_size=32 --model=Linformer
python train.py --data=ETTh1 --epochs=300 --batch_size=32 --model=LocalHBA --d_block=8
python train.py --data=ETTh1 --epochs=300 --batch_size=32 --model=HBA --d_block=8

date
