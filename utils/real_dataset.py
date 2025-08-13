import os
import cv2
import torch
import numpy as np
import random
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class RealFaceDataset(Dataset):
    """Датасет с реальными изображениями лиц"""
    
    def __init__(self, data_dir, input_size=112, augment=True, max_identities=None):
        self.data_dir = data_dir
        self.input_size = input_size
        self.augment = augment
        
        # Создание списка файлов и меток
        self.image_paths = []
        self.labels = []
        self.identity_to_label = {}
        self.label_to_identity = {}
        
        self._load_dataset(max_identities)
        
        # PyTorch трансформации
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Аугментации
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            ])
        else:
            self.augment_transform = None
    
    def _load_dataset(self, max_identities=None):
        """Загрузка датасета из папки с изображениями - каждое изображение = отдельная личность"""
        print(f"Загрузка датасета из: {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Директория {self.data_dir} не найдена")
        
        # Получаем список всех изображений
        image_files = []
        for file in os.listdir(self.data_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(file)
        
        print(f"Найдено {len(image_files)} изображений")
        
        # Ограничиваем количество изображений если нужно
        if max_identities and len(image_files) > max_identities:
            image_files = image_files[:max_identities]
            print(f"Ограничено до {max_identities} изображений")
        
        # Каждое изображение = отдельная личность
        for idx, file in enumerate(image_files):
            self.image_paths.append(os.path.join(self.data_dir, file))
            self.labels.append(idx)  # label = индекс изображения
            self.identity_to_label[idx] = idx
            self.label_to_identity[idx] = idx
        
        print(f"Загружено {len(self.image_paths)} изображений как {len(self.image_paths)} личностей")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            image = Image.new('RGB', (self.input_size, self.input_size), (128, 128, 128))
        
        # Изменение размера
        image = image.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)
        
        # Применение аугментаций
        if self.augment and self.augment_transform and random.random() > 0.3:
            image = self.augment_transform(image)
        
        # Преобразование в тензор
        image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def create_real_dataloader(data_dir, batch_size=32, input_size=112, 
                          augment=True, max_identities=None, num_workers=4, 
                          shuffle=True, val_split=0.1):
    """Создание DataLoader для реальных данных с разделением на train/val"""
    
    # Создаем полный датасет
    full_dataset = RealFaceDataset(
        data_dir=data_dir,
        input_size=input_size,
        augment=augment,
        max_identities=max_identities
    )
    
    # Разделяем на train и val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Создаем DataLoader'ы
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} изображений")
    print(f"Val: {len(val_dataset)} изображений")
    
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Тестирование датасета
    data_dir = "../data/org"
    
    print("Тестирование датасета с реальными изображениями...")
    
    # Создаем датасет
    dataset = RealFaceDataset(
        data_dir=data_dir,
        input_size=112,
        augment=True,
        max_identities=10  # Ограничиваем для теста
    )
    
    print(f"Размер датасета: {len(dataset)}")
    
    # Тест загрузки данных
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels = {labels}")
        if batch_idx >= 2:
            break
    
    print("Тестирование завершено!")
