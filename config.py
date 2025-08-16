
import os  # Для работы с операционной системой (создание директорий, пути)
from dataclasses import dataclass  # Декоратор для создания классов данных
from typing import Optional  # Тип для опциональных параметров


@dataclass
class ModelConfig:
    """Конфигурация модели - настройки архитектуры нейронной сети"""
    embedding_size: int = 512  # Размер вектора признаков (эмбеддинга) лица
    input_size: int = 112  # Размер входного изображения (112x112 пикселей)
    model_type: str = 'mobilefacenet'  # Тип модели (в данном случае MobileFaceNet)


@dataclass
class DataConfig:
    """Конфигурация данных - настройки для работы с датасетом"""
    data_dir: str = "data/extracted_images"  # Путь к папке с изображениями лиц
    num_identities: int = 1000000  # Максимальное количество уникальных личностей в датасете

    val_split: float = 0.1  # Доля данных для валидации (10% от общего количества)
    num_workers: int = 4  # Количество процессов для загрузки данных (ускоряет обучение)


@dataclass
class TrainingConfig:
    """Конфигурация обучения - параметры процесса тренировки модели"""
    batch_size: int = 512  # Размер батча (количество изображений за один проход)
    epochs: int = 50  # Количество полных проходов по всему датасету
    learning_rate: float = 1e-3  # Скорость обучения (шаг градиентного спуска)
    weight_decay: float = 5e-4  # Регуляризация весов (предотвращает переобучение)
    loss_type: str = 'arcface'  # Тип функции потерь ('arcface' или 'cosface')
    margin: float = 0.5  # Отступ для ArcFace/CosFace (увеличивает расстояние между классами)
    scale: float = 64.0  # Масштабирующий коэффициент для функции потерь
    checkpoint_dir: str = "checkpoints"  # Папка для сохранения промежуточных результатов
    save_frequency: int = 10  # Частота сохранения чекпоинтов (каждые N эпох)
    early_stopping_patience: int = 20  # Количество эпох без улучшения для ранней остановки
    validation_frequency: int = 10  # Частота валидации (каждые N эпох)
    # AMP параметры (Automatic Mixed Precision)
    use_amp: bool = True  # Использовать автоматическое смешанное представление (ускоряет обучение)


@dataclass
class InferenceConfig:
    """Конфигурация инференса - настройки для использования обученной модели"""
    model_path: str = "checkpoints/best_model.pth"  # Путь к лучшей обученной модели
    threshold: float = 0.6  # Порог для определения совпадения лиц (0-1)
    device: Optional[str] = None  # Устройство для инференса (None = автоопределение)


@dataclass
class AugmentationConfig:
    """Конфигурация аугментации - настройки для увеличения датасета"""
    horizontal_flip: bool = True  # Горизонтальное отражение изображений
    rotation_range: float = 10.0  # Диапазон поворота в градусах
    brightness_range: float = 0.2  # Диапазон изменения яркости
    contrast_range: float = 0.2  # Диапазон изменения контраста
    saturation_range: float = 0.2  # Диапазон изменения насыщенности
    hue_range: float = 0.1  # Диапазон изменения оттенка


@dataclass
class LoggingConfig:
    """Конфигурация логирования - настройки для отслеживания процесса обучения"""
    log_dir: str = "runs"  # Папка для логов TensorBoard
    tensorboard: bool = True  # Включить логирование в TensorBoard
    save_images: bool = False  # Сохранять примеры изображений
    verbose: bool = True  # Подробный вывод в консоль


class Config:
    """Основной класс конфигурации - объединяет все настройки проекта"""
    
    def __init__(self):
        # Создание экземпляров всех конфигурационных классов
        self.model = ModelConfig()  # Настройки модели
        self.data = DataConfig()  # Настройки данных
        self.training = TrainingConfig()  # Настройки обучения
        self.inference = InferenceConfig()  # Настройки инференса
        self.augmentation = AugmentationConfig()  # Настройки аугментации
        self.logging = LoggingConfig()  # Настройки логирования
    

    
    def create_directories(self):
        """Создание необходимых директорий для работы проекта"""
        # Список директорий, которые нужно создать
        directories = [
            self.training.checkpoint_dir,  # Папка для чекпоинтов
            self.logging.log_dir,  # Папка для логов
            os.path.dirname(self.inference.model_path)  # Папка для модели инференса
        ]
        # Создание каждой директории (если не существует)
        for directory in directories:
            if directory:  # Проверка, что путь не пустой
                os.makedirs(directory, exist_ok=True)  # Создать папку (exist_ok=True не вызовет ошибку, если папка уже есть)


 