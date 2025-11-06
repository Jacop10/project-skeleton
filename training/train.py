import torch

import wandb

wandb.init(project="project-skeleton", entity="s333730")

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Accuracy: {acc:.2f}%")

    wandb.log({"loss": loss, "accuracy": acc})
