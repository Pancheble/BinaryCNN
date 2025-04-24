import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from model import CNN
from config import device
from utils import transform

model_path = r'result/Epoch8_TotalLoss12.6540_Acc0.9231.pth'
image_path = r'img04.png'


def test(model_path, image_path):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        percent = probs * 100
    chihuahua = percent[0][0].item()
    muffin = percent[0][1].item()
    
    return chihuahua, muffin

if __name__ == '__main__':
    chihuahua, muffin = test(model_path=model_path, image_path=image_path)

    plt.imshow(Image.open(image_path))
    plt.title(f'chihuahua: {chihuahua}% muffin: {muffin}%')
    plt.show()