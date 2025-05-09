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

python train.py --data=ETTh1 --model=Torch --epochs=100 --batch_size=64
python train.py --data=ETTh1 --model=Base --epochs=100 --batch_size=64
python train.py --data=ETTh1 --model=LocalHBA --epochs=100 --batch_size=64 --d_block=8
python train.py --data=ETTh1 --model=HBA --epochs=100 --batch_size=64 --d_block=8
python train.py --data=ETTh1 --model=Linformer --epochs=100 --batch_size=64 --d_block=8

date
