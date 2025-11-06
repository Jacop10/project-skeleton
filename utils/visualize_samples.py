import matplotlib.pyplot as plt
import numpy as np

def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def show_samples(dataset, num_classes=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    shown_classes = []
    for img, label in dataset:
        if label not in shown_classes:
            ax = axes.flatten()[len(shown_classes)]
            ax.imshow(denormalize(img))
            ax.set_title(dataset.classes[label])
            ax.axis('off')
            shown_classes.append(label)
        if len(shown_classes) == num_classes:
            break
    plt.show()
