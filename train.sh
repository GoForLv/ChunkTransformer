# Debug
echo "Debug..."
python train.py --model=TorchTransformer --seq_len=512 --epochs=10 --train_count=1 --debug
python train.py --model=MultiHeadTransformer --seq_len=512 --epochs=10 --train_count=1 --debug
python train.py --model=MaskTransformer --seq_len=512 --epochs=10 --train_count=1 --debug
python train.py --model=ChunkTransformer --seq_len=1024 --epochs=10 --train_count=1 --debug

echo "Data: $1"
if [ "$1" = "3.27" ]; then
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