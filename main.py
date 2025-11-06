import torch
from models.custom_net import CustomNet
from dataloader.tiny_imagenet_loader import get_data_loaders
from training.train import train
from evaluation.eval import eval
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_data_loaders()

model = CustomNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0
for epoch in range(1, 6):
    train(epoch, model, train_loader, criterion, optimizer, device)
    acc = eval(model, val_loader, criterion, device)
    best_acc = max(best_acc, acc)

print(f"Best validation accuracy: {best_acc:.2f}%")
