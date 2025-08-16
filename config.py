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
    num_identities: int = 10000  # Максимальное количество изображений для обучения
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
    
    def update_from_dict(self, config_dict: dict):
        """Обновление конфигурации из словаря"""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def to_dict(self) -> dict:
        """Преобразование конфигурации в словарь"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'augmentation': self.augmentation.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_config(self, filepath: str):
        """Сохранение конфигурации в файл"""
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """Загрузка конфигурации из файла"""
        import json
        config = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config.update_from_dict(config_dict)
        return config
    
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


# Предустановленные конфигурации
def get_fast_config() -> Config:
    """Быстрая конфигурация для тестирования"""
    config = Config()
    config.training.epochs = 10
    config.training.batch_size = 32
    config.training.use_amp = True
    config.data.num_identities = 100
    config.data.samples_per_identity = 20
    return config


def get_high_accuracy_config() -> Config:
    """Конфигурация для высокой точности"""
    config = Config()
    config.training.epochs = 200
    config.training.learning_rate = 5e-4
    config.training.batch_size = 32
    config.model.embedding_size = 256
    config.data.num_identities = 2000
    config.data.samples_per_identity = 100
    return config


def get_production_config() -> Config:
    """Продакшн конфигурация"""
    config = Config()
    config.training.epochs = 150
    config.training.batch_size = 128
    config.training.learning_rate = 1e-3
    config.training.use_amp = True
    config.data.num_identities = 5000
    config.data.samples_per_identity = 50
    config.logging.verbose = False
    return config 