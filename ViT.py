import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import load_mnist_data
import matplotlib.pyplot as plt
import os

from utils import Timer
from HBA import TorchTransformer, BaseTransformer, LocalHBATransformer, HBATransformer
from Linformer import Linformer

def select_model(model_type, d_block):
    if model_type == 'Torch':
        model = TorchTransformer(
                d_model=64,
                n_head=4,
                d_ffn=256,
                num_encoder_layers=4,
                d_input=0,
                d_output=10,
                dropout=0.1
            )
    elif model_type == 'Base':
        model = BaseTransformer(
                d_model=64,
                n_head=4,
                d_ffn=256,
                num_encoder_layers=4,
                d_input=0,
                d_output=10,
                dropout=0.1
            )
    elif model_type == 'LocalHBA':
        model = LocalHBATransformer(
                d_model=64,
                n_head=4,
                d_ffn=256,
                num_encoder_layers=4,
                d_input=0,
                d_output=10,
                d_block=d_block,
                dropout=0.1
            )
    elif model_type == 'HBA':
        model = HBATransformer(
                d_model=64,
                n_head=4,
                d_ffn=256,
                num_encoder_layers=4,
                d_input=0,
                d_output=10,
                d_block=d_block,
                dropout=0.1
            )
    elif model_type == 'Linformer':
        model = Linformer(
                d_model=64,
                n_head=4,
                d_ffn=256,
                num_encoder_layers=4,
                seq_len=64,
                d_input=0,
                d_output=10,
                dropout=0.1
            )
    return model

# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(dataloader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

# 测试函数
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(dataloader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

def visualize(train_losses, test_losses, train_accs, test_accs, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(
        'log',
        'ViT',
        f'{save_path}.png'
    ))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='帮助文档')

    parser.add_argument('--data', help='MNIST')
    parser.add_argument('--model', help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--d_block', type=int, help='')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = select_model(args.model, args.d_block).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    train_dataset, test_dataset = load_mnist_data()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    timer = Timer()
    log_path = os.path.join(
        'log',
        'ViT',
        'ViT.txt'
    )
    log = open(log_path, 'a', encoding='utf-8')

    print(args)
    log.write(str(args))

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        log.write(f"\nEpoch {epoch+1}/{args.epochs}\n")
        
        # 训练
        timer.start('train')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        timer.stop('train')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 测试
        timer.start('test')
        test_loss, test_acc = test(model, test_loader, criterion, device)
        timer.stop('test')

        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        log.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")
        log.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")
        log.write(timer.display_record('epoch'))

    log.write(timer.display_record('average'))
    visualize(train_losses, test_losses, train_accs, test_accs, save_path=f'{args.model}-{args.batch_size}')