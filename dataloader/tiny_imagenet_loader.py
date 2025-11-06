import os
import shutil
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# --- DAL LAB2: creazione della struttura di val/ ---
def prepare_val_folder(val_dir):
    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            src = os.path.join(val_dir, 'images', fn)
            dst = os.path.join(val_dir, cls, fn)
            shutil.copyfile(src, dst)
    shutil.rmtree(os.path.join(val_dir, 'images'))

# --- DAL LAB1/LAB2: trasformazioni e caricamento dataset ---
def get_data_loaders(data_dir='data/tiny-imagenet-200', batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
