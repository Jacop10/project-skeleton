import torch

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return acc
