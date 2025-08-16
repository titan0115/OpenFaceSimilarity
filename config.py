import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    embedding_size: int = 512  # Исправлено с 128 на 512
    input_size: int = 112
    model_type: str = 'mobilefacenet'


@dataclass
class DataConfig:
    """Конфигурация данных"""
    data_dir: str = "data/pre"  # Директория с предобработанными данными
    num_identities: int = 100000  # Максимальное количество изображений для обучения
    samples_per_identity: int = 50  # Количество сэмплов на идентичность
    val_split: float = 0.1
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    batch_size: int = 512
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    loss_type: str = 'arcface'  # 'arcface' или 'cosface'
    margin: float = 0.5
    scale: float = 64.0
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10
    early_stopping_patience: int = 20
    validation_frequency: int = 10  # Частота валидации (каждые N эпох)
    # AMP параметры
    use_amp: bool = True  # Использовать Automatic Mixed Precision


@dataclass
class InferenceConfig:
    """Конфигурация инференса"""
    model_path: str = "checkpoints/best_model.pth"
    threshold: float = 0.6
    device: Optional[str] = None  # None для автоопределения


@dataclass
class AugmentationConfig:
    """Конфигурация аугментации"""
    horizontal_flip: bool = True
    rotation_range: float = 10.0
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    saturation_range: float = 0.2
    hue_range: float = 0.1


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    log_dir: str = "runs"
    tensorboard: bool = True
    save_images: bool = False
    verbose: bool = True


class Config:
    """Основной класс конфигурации"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.augmentation = AugmentationConfig()
        self.logging = LoggingConfig()
    

    
    def create_directories(self):
        """Создание необходимых директорий"""
        directories = [
            self.training.checkpoint_dir,
            self.logging.log_dir,
            os.path.dirname(self.inference.model_path)
        ]
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)


 