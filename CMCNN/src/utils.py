
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
from config import device
from config import config


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder(root=r'dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root=r'dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])


def Validation_Accuracy(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / len(val_loader.dataset)
    return val_accuracy