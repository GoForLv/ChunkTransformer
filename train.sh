python train.py --data=ETTh1 --model=Torch --epochs=5 --batch_size=64 --seq_len=128
python train.py --data=ETTh1 --model=Base --epochs=5 --batch_size=64 --seq_len=128
python train.py --data=ETTh1 --model=LocalHBA --epochs=5 --batch_size=64 --d_block=8 --seq_len=128
python train.py --data=ETTh1 --model=HBA --epochs=5 --batch_size=64 --d_block=8 --seq_len=128
python train.py --data=ETTh1 --model=Linformer --epochs=5 --batch_size=64 --d_block=8 --seq_len=128
