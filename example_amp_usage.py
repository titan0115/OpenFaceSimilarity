#!/usr/bin/env python3
"""
Пример использования AMP (Automatic Mixed Precision) в проекте распознавания лиц
"""

import torch
from config import Config, get_fast_config, get_production_config
from train import FaceRecognitionTrainer

def demo_amp_config():
    """Демонстрация настройки AMP через конфигурацию"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ НАСТРОЙКИ AMP")
    print("=" * 60)
    
    # Создание конфигурации с AMP
    config = Config()
    
    # Настройка AMP параметров
    config.training.use_amp = True
    config.training.amp_opt_level = "O1"  # Оптимальный баланс производительности/точности
    config.training.amp_loss_scale = None  # Автоматическое масштабирование
    
    # Печать настроек
    print(f"AMP включен: {config.training.use_amp}")
    print(f"Уровень оптимизации: {config.training.amp_opt_level}")
    print(f"Масштабирование loss: {config.training.amp_loss_scale}")
    
    # Предустановленные конфигурации с AMP
    print("\n--- Быстрая конфигурация ---")
    fast_config = get_fast_config()
    print(f"AMP: {fast_config.training.use_amp}")
    
    print("\n--- Продакшн конфигурация ---")
    prod_config = get_production_config()
    print(f"AMP: {prod_config.training.use_amp}")
    
    return config

def demo_amp_levels():
    """Демонстрация различных уровней AMP оптимизации"""
    print("\n" + "=" * 60)
    print("УРОВНИ AMP ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    levels = {
        "O0": "FP32 training (без AMP)",
        "O1": "Conservative Mixed Precision (рекомендуемый)",
        "O2": "Fast Mixed Precision",
        "O3": "FP16 training (экспериментальный)"
    }
    
    for level, description in levels.items():
        print(f"{level}: {description}")
    
    print("\nРекомендации:")
    print("- O1: Лучший баланс между скоростью и стабильностью")
    print("- O2: Максимальная скорость, может потребовать настройки loss_scale")
    print("- O0: Отключает AMP полностью")
    print("- O3: Экспериментальный, не рекомендуется для продакшна")

def demo_amp_benefits():
    """Демонстрация преимуществ AMP"""
    print("\n" + "=" * 60)
    print("ПРЕИМУЩЕСТВА AMP")
    print("=" * 60)
    
    benefits = [
        "🚀 Ускорение обучения до 1.5-2x на современных GPU",
        "💾 Снижение потребления видеопамяти примерно на 50%",
        "📈 Возможность использования больших batch_size",
        "⚡ Автоматическое управление точностью вычислений",
        "🛡️ Защита от переполнения градиентов через loss scaling",
        "🔧 Простота интеграции - минимальные изменения кода"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    print("\nТребования:")
    print("- GPU с поддержкой Tensor Cores (RTX 20xx+, V100+)")
    print("- PyTorch 1.6+ с поддержкой CUDA")
    print("- CUDA 10.1+")

def main():
    """Основная функция демонстрации"""
    print("Демонстрация использования AMP в проекте распознавания лиц")
    
    # Демонстрация настроек
    config = demo_amp_config()
    
    # Демонстрация уровней оптимизации
    demo_amp_levels()
    
    # Демонстрация преимуществ
    demo_amp_benefits()
    
    # Проверка доступности CUDA
    print("\n" + "=" * 60)
    print("ПРОВЕРКА СИСТЕМЫ")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступна: {cuda_available}")
    
    if cuda_available:
        print(f"Версия CUDA: {torch.version.cuda}")
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Название GPU: {torch.cuda.get_device_name(0)}")
        
        # Проверка поддержки AMP
        try:
            from torch.cuda.amp import autocast, GradScaler
            print("✅ AMP полностью поддерживается")
        except ImportError:
            print("❌ AMP не поддерживается в данной версии PyTorch")
    else:
        print("⚠️ CUDA недоступна - AMP будет автоматически отключен")
    
    print("\n" + "=" * 60)
    print("ПРИМЕР ЗАПУСКА С AMP")
    print("=" * 60)
    print("python train.py --use_amp --amp_opt_level O1 --batch_size 128")
    print("python train.py --use_amp --amp_opt_level O2 --batch_size 256")

if __name__ == "__main__":
    main()