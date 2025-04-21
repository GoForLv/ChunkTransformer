#!/bin/bash
#SBATCH -J pytorch_test
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

ulimit -u unlimited
export PATH="/share/apps/anaconda3/bin:$PATH"
source activate pytorch-env

date

echo "Train..."
python train.py --data=ETTh1 --model=Torch --epochs=50 --batch_size=32
python train.py --data=ETTh1 --model=Origin --epochs=50 --batch_size=32

python train.py --data=ETTh1 --model=Chunk --epochs=50 --batch_size=32 --d_chunk=8
python train.py --data=ETTh1 --model=Chunk --epochs=50 --batch_size=32 --d_chunk=16
python train.py --data=ETTh1 --model=Chunk --epochs=50 --batch_size=32 --d_chunk=32
python train.py --data=ETTh1 --model=Chunk --epochs=50 --batch_size=32 --d_chunk=64
python train.py --data=ETTh1 --model=Chunk --epochs=50 --batch_size=32 --d_chunk=0

date
