python ViT.py --data=MNIST --epochs=1 --batch_size=64 --model=Base
python ViT.py --data=MNIST --epochs=1 --batch_size=64 --model=Torch
python ViT.py --data=MNIST --epochs=1 --batch_size=64 --model=Linformer
python ViT.py --data=MNIST --epochs=1 --batch_size=64 --model=LocalHBA --d_block=8
python ViT.py --data=MNIST --epochs=1 --batch_size=64 --model=HBA --d_block=8

python ViT.py --data=MNIST --epochs=1 --batch_size=32 --model=Base
python ViT.py --data=MNIST --epochs=1 --batch_size=32 --model=Torch
python ViT.py --data=MNIST --epochs=1 --batch_size=32 --model=Linformer
python ViT.py --data=MNIST --epochs=1 --batch_size=32 --model=LocalHBA --d_block=8
python ViT.py --data=MNIST --epochs=1 --batch_size=32 --model=HBA --d_block=8