import os
import cv2
import torch
import numpy as np
import random
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import urllib.request
from tqdm import tqdm


class FaceAugmentation:
    """Класс для аугментации изображений лиц"""
    
    def __init__(self, input_size=112):
        self.input_size = input_size
        
    def geometric_augmentation(self, image):
        """Геометрические преобразования"""
        # Случайный поворот
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Случайное масштабирование
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            
            # Обрезка или дополнение до исходного размера
            if new_h > h or new_w > w:
                y_start = (new_h - h) // 2 if new_h > h else 0
                x_start = (new_w - w) // 2 if new_w > w else 0
                image = image[y_start:y_start+h, x_start:x_start+w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = cv2.copyMakeBorder(image, pad_h, h-new_h-pad_h, 
                                         pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)
        
        # Случайный сдвиг
        if random.random() > 0.5:
            h, w = image.shape[:2]
            shift_x = random.randint(-int(w*0.1), int(w*0.1))
            shift_y = random.randint(-int(h*0.1), int(h*0.1))
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, M, (w, h))
        
        return image
    
    def color_augmentation(self, image):
        """Цветовые преобразования"""
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Яркость
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image_pil)
            image_pil = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Контрастность
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image_pil)
            image_pil = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Насыщенность
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(image_pil)
            image_pil = enhancer.enhance(random.uniform(0.8, 1.2))
        
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return image
    
    def noise_augmentation(self, image):
        """Добавление шума"""
        if random.random() > 0.7:
            # Гауссов шум
            noise = np.random.normal(0, random.uniform(5, 15), image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def lighting_augmentation(self, image):
        """Имитация различных условий освещения"""
        if random.random() > 0.6:
            # Случайное затемнение/осветление части изображения
            h, w = image.shape[:2]
            overlay = np.ones_like(image) * random.randint(30, 100)
            
            # Создание маски градиента
            x_center = random.randint(w//4, 3*w//4)
            y_center = random.randint(h//4, 3*h//4)
            
            y, x = np.ogrid[:h, :w]
            mask = ((x - x_center)**2 + (y - y_center)**2) < (min(h, w)//2)**2
            
            alpha = 0.3
            image[mask] = cv2.addWeighted(image[mask], 1-alpha, overlay[mask], alpha, 0)
        
        return image
    
    def apply_augmentations(self, image):
        """Применение всех аугментаций"""
        image = self.geometric_augmentation(image)
        image = self.color_augmentation(image)
        image = self.noise_augmentation(image)
        image = self.lighting_augmentation(image)
        
        # Финальное изменение размера
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        return image


def create_dataloader(data_dir, batch_size=32, num_workers=4, shuffle=True, 
                     input_size=112, augment=True, max_identities=None, val_split=0.1):
    """Создание DataLoader для обучения с реальными данными"""
    
    from .real_dataset import RealFaceDataset, create_real_dataloader
    
    # Используем реальные данные
    train_dataloader, val_dataloader = create_real_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        input_size=input_size,
        augment=augment,
        max_identities=max_identities,
        num_workers=num_workers,
        shuffle=shuffle,
        val_split=val_split
    )
    
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Тестирование генератора датасета
    data_dir = "../data/org"
    
    print("Тестирование датасета с реальными данными...")
    
    # Тест создания DataLoader
    train_loader, val_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=4,
        max_identities=10,
        num_workers=0,  # Для теста без многопроцессности
        shuffle=True
    )
    
    print(f"Train loader: {len(train_loader.dataset)} изображений")
    print(f"Val loader: {len(val_loader.dataset)} изображений")
    
    # Тест загрузки данных
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels = {labels}")
        if batch_idx >= 2:
            break
    
    print("Тестирование завершено!")