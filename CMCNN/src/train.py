import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import CNN
from config import config
from config import device
from utils import val_loader, train_loader
from utils import Validation_Accuracy

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

for epoch in range(config['epochs']):
    model.train()
    correct = 0
    running_loss = 0.0

    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{config['epochs']}]")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item())

    accuracy = correct / len(train_loader.dataset)
    torch.save(model.state_dict(), f'result/Epoch{epoch+1}_TotalLoss{running_loss:.4f}_Acc{Validation_Accuracy(model=model):.4f}.pth')
    print(f"Epoch[{epoch+1}/{config['epochs']}] Total Loss: {running_loss:.4f} Loss: {loss.item():.4f}| Train Accuracy: {accuracy:.4f} Validation_Accuracy: {Validation_Accuracy(model=model):.4f}")