from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from PIL import Image
import os

def download_data(path):
    data = []
    y = []
    for path_image in sorted(os.listdir(path=path)):
        image = Image.open(path + path_image).resize((180, 180))
        image = np.array(image)
        data.append(image.astype(np.uint8))
        if 'cat' in path_image:
            y.append(0)
        else:
            y.append(1)
    return np.array(data), np.array(y)



def prepare_dataloaders(train_dataset, y_train, test_dataset, y_test, batch_size):
    train_tensor = torch.tensor(train_dataset / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    test_tensor = torch.tensor(test_dataset / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_tensor_dataset = TensorDataset(train_tensor, y_train_tensor)
    test_tensor_dataset = TensorDataset(test_tensor, y_test_tensor)
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size)
    return train_loader, test_loader