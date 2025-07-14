import torch

from data_loader import download_data, prepare_dataloaders
from model import CNNBinaryClassifier
from train import train_model

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

def run(apply_augmentation):
    train_dataset, y_train = download_data(r"./data/Train/")
    test_dataset, y_test = download_data(r"./data/Test/")
    # гипер-параметры обучения
    learning_rate = 0.0001 # скорость обучения
    num_epochs = 31 # количество эпох
    batch_size = 32 # размер батча

    train_loader, test_loader = prepare_dataloaders(train_dataset, y_train, test_dataset, y_test, batch_size)
    model = CNNBinaryClassifier()
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, test_loader,
                                                                             num_epochs=num_epochs, learning_rate=learning_rate, apply_augmentation=apply_augmentation)

    test_loss = val_losses[-1]
    test_accuracy = val_accuracies[-1]
    print(f"\nРезультаты на тестовой выборке:\nПотери: {test_loss:.4f}, Точность: {test_accuracy:.4f}")

run(False)
run(True)