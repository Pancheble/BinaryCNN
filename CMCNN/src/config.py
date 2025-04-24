import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001
}