# Debug
python train.py --model=TorchTransformer --seq_len=16 --epochs=2 --train_count=1 --debug
python train.py --model=MultiHeadTransformer --seq_len=16 --epochs=2 --train_count=1 --debug
python train.py --model=MaskTransformer --seq_len=16 --epochs=2 --train_count=1 --debug
python train.py --model=ChunkTransformer --seq_len=16 --epochs=2 --train_count=1 --debug

# 3.27 阿里云实验脚本
# python train.py --model=MultiHeadTransformer --seq_len=256 --epochs=50 --train_count=3
# python train.py --model=MultiHeadTransformer --seq_len=512 --epochs=50 --train_count=3

# python train.py --model=ChunkTransformer --seq_len=256 --epochs=50 --train_count=3
# python train.py --model=ChunkTransformer --seq_len=512 --epochs=50 --train_count=3

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