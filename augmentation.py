from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.ToPILImage(), # нужно, если вход — numpy-изображение
    transforms.RandomHorizontalFlip(p=0.5), # Аналог RandomFlip
    transforms.RandomRotation(degrees=36), # Аналог RandomRotation(0.1)
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)), # Аналог RandomZoom(0.2)
    transforms.Resize((180, 180)), # Обязательно
    transforms.ToTensor()
])

# Тестовые данные — без аугментации
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((180, 180)),
    transforms.ToTensor()
])