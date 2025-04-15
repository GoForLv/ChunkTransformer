echo "Date: $1"
if [ "$1" = "4.15" ]; then
    # python train.py --data=ETTh1 --model=Origin --epochs=3 --batch_size=32 --seq_len=128
    # python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=128
    # python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=64
    # python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=32
    # python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=16
    # python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=8
    python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=4
    python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=2
    python train.py --data=ETTh1 --model=Chunk --epochs=3 --batch_size=32 --seq_len=128 --d_chunk=1

elif [ "$1" = "4.11" ]; then
    # # Debug
    # echo "Debug..."
    # python train.py --data=ETTh1 --model=Torch --epochs=10 --batch_size=32
    # python train.py --data=ETTh1 --model=Chunk --epochs=10 --batch_size=32 --d_chunk=0 

    # train
    echo "Train..."
    python train.py --data=ETTh1 --model=Torch --epochs=30 --batch_size=32
    python train.py --data=ETTh1 --model=Origin --epochs=30 --batch_size=32

    python train.py --data=ETTh1 --model=Chunk --epochs=30 --batch_size=32 --d_chunk=8 
    python train.py --data=ETTh1 --model=Chunk --epochs=30 --batch_size=32 --d_chunk=16
    python train.py --data=ETTh1 --model=Chunk --epochs=30 --batch_size=32 --d_chunk=32
    python train.py --data=ETTh1 --model=Chunk --epochs=30 --batch_size=32 --d_chunk=64
    python train.py --data=ETTh1 --model=Chunk --epochs=30 --batch_size=32 --d_chunk=0 
elif [ "$1" = "4.9" ]; then
    python train.py --data=ETTh1 --model=TorchTransformer --seq_len=128 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=TorchTransformer --seq_len=256 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=TorchTransformer --seq_len=512 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=TorchTransformer --seq_len=1024 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=TorchTransformer --seq_len=2048 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=TorchTransformer --seq_len=4096 --epochs=50 --train_count=3

    python train.py --data=ETTh1 --model=MultiHeadTransformer --seq_len=128 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=MultiHeadTransformer --seq_len=256 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=MultiHeadTransformer --seq_len=512 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=MultiHeadTransformer --seq_len=1024 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=MultiHeadTransformer --seq_len=2048 --epochs=50 --train_count=3
    python train.py --data=ETTh1 --model=MultiHeadTransformer --seq_len=4096 --epochs=50 --train_count=3

    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=128 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=256 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=512 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=1024 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=2048 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=4096 --epochs=50 --d_chunk=16 --train_count=3

    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=128 --epochs=50 --d_chunk=32 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=256 --epochs=50 --d_chunk=32 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=512 --epochs=50 --d_chunk=32 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=1024 --epochs=50 --d_chunk=32 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=2048 --epochs=50 --d_chunk=32 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=4096 --epochs=50 --d_chunk=32 --train_count=3

    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=128 --epochs=50 --d_chunk=64 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=256 --epochs=50 --d_chunk=64 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=512 --epochs=50 --d_chunk=64 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=1024 --epochs=50 --d_chunk=64 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=2048 --epochs=50 --d_chunk=64 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=4096 --epochs=50 --d_chunk=64 --train_count=3

    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=128 --epochs=50 --d_chunk=0 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=256 --epochs=50 --d_chunk=0 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=512 --epochs=50 --d_chunk=0 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=1024 --epochs=50 --d_chunk=0 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=2048 --epochs=50 --d_chunk=0 --train_count=3
    python train.py --data=ETTh1 --model=ChunkTransformer --seq_len=4096 --epochs=50 --d_chunk=0 --train_count=3
elif [ "$1" = "3.27" ]; then
    # 3.27 阿里云实验脚本
    python train.py --model=MultiHeadTransformer --seq_len=256 --epochs=50 --train_count=3
    python train.py --model=MultiHeadTransformer --seq_len=512 --epochs=50 --train_count=3

    python train.py --model=ChunkTransformer --seq_len=256 --epochs=50 --train_count=3
    python train.py --model=ChunkTransformer --seq_len=512 --epochs=50 --train_count=3
elif [ "$1" = "3.28" ]; then
    # 3.28 阿里云实验脚本
    python train.py --model=MultiHeadTransformer --seq_len=256 --epochs=50 --train_count=1
    python train.py --model=MultiHeadTransformer --seq_len=512 --epochs=50 --train_count=1
    python train.py --model=MultiHeadTransformer --seq_len=1024 --epochs=50 --train_count=3

    python train.py --model=ChunkTransformer --seq_len=256 --epochs=50 --train_count=1
    python train.py --model=ChunkTransformer --seq_len=512 --epochs=50 --train_count=1
    python train.py --model=ChunkTransformer --seq_len=1024 --epochs=50 --train_count=3

    python train.py --model=MaskTransformer --seq_len=256 --epochs=50 --train_count=3
    python train.py --model=MaskTransformer --seq_len=512 --epochs=50 --train_count=3
    python train.py --model=MaskTransformer --seq_len=1024 --epochs=50 --train_count=3
elif [ "$1" = "3.30" ]; then
    # 3.30 阿里云实验脚本
    python train.py --model=MultiHeadTransformer --seq_len=128 --epochs=50 --train_count=3
    python train.py --model=MultiHeadTransformer --seq_len=256 --epochs=50 --train_count=3
    python train.py --model=MultiHeadTransformer --seq_len=512 --epochs=50 --train_count=3
    python train.py --model=MultiHeadTransformer --seq_len=640 --epochs=50 --train_count=3
    # python train.py --model=MultiHeadTransformer --seq_len=1024 --epochs=50 --train_count=3

    python train.py --model=ChunkTransformer --seq_len=128 --epochs=50 --d_chunk=8 --train_count=3
    python train.py --model=ChunkTransformer --seq_len=256 --epochs=50 --d_chunk=8 --train_count=3
    python train.py --model=ChunkTransformer --seq_len=512 --epochs=50 --d_chunk=8 --train_count=3
    python train.py --model=ChunkTransformer --seq_len=640 --epochs=50 --d_chunk=8 --train_count=3
    # python train.py --model=ChunkTransformer --seq_len=1024 --epochs=50 --d_chunk=8 --train_count=3

    python train.py --model=ChunkTransformer --seq_len=128 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --model=ChunkTransformer --seq_len=256 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --model=ChunkTransformer --seq_len=512 --epochs=50 --d_chunk=16 --train_count=3
    python train.py --model=ChunkTransformer --seq_len=640 --epochs=50 --d_chunk=16 --train_count=3
else
    echo "Unknown Data: $1"
    exit 1
fi