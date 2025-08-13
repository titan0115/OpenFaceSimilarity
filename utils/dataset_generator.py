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


class SyntheticFaceDataset(Dataset):
    """Датасет с синтетическими лицами и аугментациями"""
    
    def __init__(self, data_dir, num_identities=1000, samples_per_identity=50, 
                 input_size=112, augment=True):
        self.data_dir = data_dir
        self.num_identities = num_identities
        self.samples_per_identity = samples_per_identity
        self.input_size = input_size
        self.augment = augment
        
        self.augmenter = FaceAugmentation(input_size) if augment else None
        
        # Создание списка файлов и меток
        self.image_paths = []
        self.labels = []
        
        self._create_dataset_structure()
        
        # PyTorch трансформации
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _create_dataset_structure(self):
        """Создание структуры датасета"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Если датасет уже существует, загружаем его
        if os.path.exists(os.path.join(self.data_dir, 'dataset_info.txt')):
            self._load_existing_dataset()
        else:
            self._generate_synthetic_dataset()
    
    def _load_existing_dataset(self):
        """Загрузка существующего датасета"""
        print("Загрузка существующего датасета...")
        for identity_id in range(self.num_identities):
            identity_dir = os.path.join(self.data_dir, f"identity_{identity_id:04d}")
            if os.path.exists(identity_dir):
                for img_file in os.listdir(identity_dir):
                    if img_file.endswith(('.jpg', '.png')):
                        self.image_paths.append(os.path.join(identity_dir, img_file))
                        self.labels.append(identity_id)
    
    def _generate_synthetic_dataset(self):
        """Генерация синтетического датасета с базовыми лицами и аугментациями"""
        print("Генерация синтетического датасета...")
        
        # Создание базовых синтетических лиц (простые геометрические формы)
        for identity_id in tqdm(range(self.num_identities), desc="Создание личностей"):
            identity_dir = os.path.join(self.data_dir, f"identity_{identity_id:04d}")
            os.makedirs(identity_dir, exist_ok=True)
            
            # Генерация базового лица для данной личности
            base_face = self._generate_base_face(identity_id)
            
            # Создание множественных вариаций
            for sample_id in range(self.samples_per_identity):
                # Применение аугментаций к базовому лицу
                if self.augmenter:
                    augmented_face = self.augmenter.apply_augmentations(base_face.copy())
                else:
                    augmented_face = cv2.resize(base_face, (self.input_size, self.input_size))
                
                # Сохранение изображения
                img_path = os.path.join(identity_dir, f"sample_{sample_id:03d}.jpg")
                cv2.imwrite(img_path, augmented_face)
                
                self.image_paths.append(img_path)
                self.labels.append(identity_id)
        
        # Сохранение информации о датасете
        with open(os.path.join(self.data_dir, 'dataset_info.txt'), 'w') as f:
            f.write(f"Identities: {self.num_identities}\n")
            f.write(f"Samples per identity: {self.samples_per_identity}\n")
            f.write(f"Total samples: {len(self.image_paths)}\n")
    
    def _generate_base_face(self, identity_id):
        """Генерация базового синтетического лица"""
        # Используем identity_id как сид для воспроизводимости
        np.random.seed(identity_id)
        
        # Создание изображения 128x128
        face = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Цвет кожи (вариации)
        skin_colors = [
            (220, 180, 140),  # светлый
            (200, 160, 120),  # средний
            (180, 140, 100),  # темный
        ]
        skin_color = skin_colors[identity_id % len(skin_colors)]
        
        # Лицо (овал)
        center = (64, 64)
        face_width = random.randint(35, 45)
        face_height = random.randint(45, 55)
        cv2.ellipse(face, center, (face_width, face_height), 0, 0, 360, skin_color, -1)
        
        # Глаза
        eye_color = (random.randint(50, 100), random.randint(50, 100), random.randint(50, 100))
        left_eye = (center[0] - 15, center[1] - 10)
        right_eye = (center[0] + 15, center[1] - 10)
        cv2.circle(face, left_eye, 6, (255, 255, 255), -1)
        cv2.circle(face, right_eye, 6, (255, 255, 255), -1)
        cv2.circle(face, left_eye, 3, eye_color, -1)
        cv2.circle(face, right_eye, 3, eye_color, -1)
        
        # Нос
        nose_points = np.array([
            [center[0], center[1] + 5],
            [center[0] - 3, center[1] + 15],
            [center[0] + 3, center[1] + 15]
        ], np.int32)
        cv2.fillPoly(face, [nose_points], (int(skin_color[0]*0.9), 
                                          int(skin_color[1]*0.9), 
                                          int(skin_color[2]*0.9)))
        
        # Рот
        mouth_color = (random.randint(150, 200), random.randint(50, 100), random.randint(50, 100))
        cv2.ellipse(face, (center[0], center[1] + 25), (8, 4), 0, 0, 360, mouth_color, -1)
        
        # Волосы
        hair_color = (random.randint(20, 100), random.randint(20, 100), random.randint(20, 100))
        cv2.ellipse(face, (center[0], center[1] - 20), (face_width + 5, 25), 0, 180, 360, hair_color, -1)
        
        # Сброс сида
        np.random.seed()
        
        return face
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Дополнительная аугментация во время обучения
        if self.augment and self.augmenter and random.random() > 0.3:
            image = self.augmenter.apply_augmentations(image)
        
        # Преобразование в тензор
        image = Image.fromarray(image)
        image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def create_dataloader(data_dir, batch_size=32, num_identities=1000, 
                     samples_per_identity=50, num_workers=4, shuffle=True, use_real_data=False):
    """Создание DataLoader для обучения"""
    
    if use_real_data:
        # Используем реальные данные
        from .real_dataset import RealFaceDataset
        
        dataset = RealFaceDataset(
            data_dir=data_dir,
            input_size=112,
            augment=True,
            max_identities=num_identities
        )
    else:
        # Используем синтетические данные
        dataset = SyntheticFaceDataset(
            data_dir=data_dir,
            num_identities=num_identities,
            samples_per_identity=samples_per_identity,
            augment=True
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Тестирование генератора датасета
    data_dir = "../data/synthetic_faces"
    
    print("Создание тестового датасета...")
    dataset = SyntheticFaceDataset(
        data_dir=data_dir,
        num_identities=10,
        samples_per_identity=5,
        augment=True
    )
    
    print(f"Размер датасета: {len(dataset)}")
    
    # Тест загрузки данных
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels = {labels}")
        if batch_idx >= 2:
            break
    
    print("Тестирование завершено!")