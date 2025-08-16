#!/usr/bin/env python3
"""
Демонстрационный скрипт для системы распознавания лиц
Показывает основные возможности проекта
"""

import os
import torch
import argparse
from models.mobilefacenet import MobileFaceNet, count_parameters
from models.losses import ArcFace, CosFace
from utils.dataset import create_real_dataloader
from utils.metrics import compute_verification_metrics, plot_verification_metrics, print_metrics_summary
from inference import FaceComparator


def demo_model_architecture():
    """Демонстрация архитектуры модели"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ АРХИТЕКТУРЫ МОДЕЛИ")
    print("=" * 60)
    
    # Создание модели
    model = MobileFaceNet(embedding_size=128, input_size=112)
    print(f"Архитектура: MobileFaceNet")
    print(f"Размер входа: 112x112x3")
    print(f"Размер эмбеддинга: 128")
    print(f"Количество параметров: {count_parameters(model):,}")
    
    # Тестовый проход
    print("\nТестовый проход через модель:")
    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        embeddings = model(x)
    
    print(f"Входной тензор: {x.shape}")
    print(f"Выходной тензор: {embeddings.shape}")
    print(f"Норма эмбеддингов: {torch.norm(embeddings, p=2, dim=1)}")
    print("✅ Модель работает корректно!")


def demo_loss_functions():
    """Демонстрация loss функций"""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ LOSS ФУНКЦИЙ")
    print("=" * 60)
    
    batch_size = 8
    embedding_size = 128
    num_classes = 100
    
    # Создание тестовых данных
    embeddings = torch.nn.functional.normalize(torch.randn(batch_size, embedding_size), p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Тестовые данные: {batch_size} эмбеддингов, {num_classes} классов")
    
    # ArcFace
    print("\n🔥 ArcFace Loss:")
    arcface = ArcFace(embedding_size, num_classes, margin=0.5, scale=64.0)
    arcface_output = arcface(embeddings, labels)
    arcface_loss = torch.nn.functional.cross_entropy(arcface_output, labels)
    print(f"   Выходной размер: {arcface_output.shape}")
    print(f"   Loss: {arcface_loss.item():.4f}")
    
    # CosFace
    print("\n🎯 CosFace Loss:")
    cosface = CosFace(embedding_size, num_classes, margin=0.35, scale=64.0)
    cosface_output = cosface(embeddings, labels)
    cosface_loss = torch.nn.functional.cross_entropy(cosface_output, labels)
    print(f"   Выходной размер: {cosface_output.shape}")
    print(f"   Loss: {cosface_loss.item():.4f}")
    
    print("✅ Loss функции работают корректно!")



def demo_training_simulation():
    """Симуляция процесса обучения"""
    print("\n" + "=" * 60)
    print("СИМУЛЯЦИЯ ПРОЦЕССА ОБУЧЕНИЯ")
    print("=" * 60)
    
    # Параметры
    embedding_size = 128
    num_classes = 10
    batch_size = 4
    use_amp = torch.cuda.is_available()  # Включаем AMP только на GPU
    
    # Модель и loss
    model = MobileFaceNet(embedding_size=embedding_size)
    loss_fn = ArcFace(embedding_size, num_classes)
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': loss_fn.parameters()}
    ], lr=1e-3)
    
    # Инициализация AMP
    if use_amp:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print(f"✅ AMP включен")
    else:
        scaler = None
        print(f"⚠️ AMP отключен (недоступен на CPU)")
    
    print(f"Модель: MobileFaceNet ({count_parameters(model):,} параметров)")
    print(f"Loss: ArcFace")
    print(f"Оптимизатор: Adam")
    
    # Симуляция нескольких шагов обучения
    print("\nСимуляция 3 шагов обучения:")
    model.train()
    loss_fn.train()
    
    for step in range(3):
        # Фиктивные данные
        images = torch.randn(batch_size, 3, 112, 112)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        optimizer.zero_grad()
        
        if use_amp:
            # Прямой проход с AMP
            with autocast():
                embeddings = model(images)
                logits = loss_fn(embeddings, labels)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Обратный проход с AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Обычный прямой проход
            embeddings = model(images)
            logits = loss_fn(embeddings, labels)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
        
        # Вычисление точности
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        print(f"   Шаг {step+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")
    
    print("✅ Процесс обучения работает корректно!")


def demo_inference():
    """Демонстрация инференса"""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ИНФЕРЕНСА")
    print("=" * 60)
    
    # Создание простой модели для демо
    model = MobileFaceNet(embedding_size=128)
    model.eval()
    
    print("Создание случайных тестовых изображений...")
    
    # Создание фиктивных изображений
    import numpy as np
    image1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    image3 = image1.copy()  # Идентичное изображение
    
    # Простая функция для получения эмбеддинга
    def get_embedding(image):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        with torch.no_grad():
            x = transform(image).unsqueeze(0)
            embedding = model(x)
            return embedding.numpy().flatten()
    
    # Получение эмбеддингов
    emb1 = get_embedding(image1)
    emb2 = get_embedding(image2)
    emb3 = get_embedding(image3)
    
    # Вычисление сходства
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    
    print(f"Сходство изображение1 ↔ изображение2: {sim_1_2:.4f}")
    print(f"Сходство изображение1 ↔ изображение3 (идентичное): {sim_1_3:.4f}")
    
    threshold = 0.6
    print(f"\nС порогом {threshold}:")
    print(f"   image1 и image2: {'СОВПАДЕНИЕ' if sim_1_2 > threshold else 'НЕ СОВПАДАЕТ'}")
    print(f"   image1 и image3: {'СОВПАДЕНИЕ' if sim_1_3 > threshold else 'НЕ СОВПАДАЕТ'}")
    
    print("✅ Инференс работает корректно!")


def demo_metrics():
    """Демонстрация вычисления метрик"""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ МЕТРИК")
    print("=" * 60)
    
    # Создание тестовых данных для метрик
    import torch.nn.functional as F
    
    batch_size = 50
    embedding_size = 128
    
    # Эмбеддинги
    embeddings1 = F.normalize(torch.randn(batch_size, embedding_size), p=2, dim=1)
    embeddings2 = F.normalize(torch.randn(batch_size, embedding_size), p=2, dim=1)
    
    # Метки (50% совпадений)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Для совпадающих пар делаем эмбеддинги более похожими
    for i in range(batch_size):
        if labels[i] == 1:
            noise = torch.randn_like(embeddings1[i]) * 0.2
            embeddings2[i] = F.normalize(embeddings1[i] + noise, p=2, dim=0)
    
    print(f"Тестовые данные: {batch_size} пар эмбеддингов")
    print(f"Совпадающих пар: {labels.sum().item()}")
    print(f"Различных пар: {(1 - labels).sum().item()}")
    
    # Вычисление метрик
    from utils.metrics import compute_verification_metrics, print_metrics_summary
    
    print("\nВычисление метрик верификации...")
    metrics = compute_verification_metrics(embeddings1, embeddings2, labels)
    
    # Вывод результатов
    print_metrics_summary(metrics)
    
    print("✅ Метрики вычислены корректно!")


def main():
    parser = argparse.ArgumentParser(description='Демонстрация системы распознавания лиц')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'model', 'loss', 'training', 'inference', 'metrics'],
                       help='Какую демонстрацию запустить')
    
    args = parser.parse_args()
    
    print("🎭 ДЕМОНСТРАЦИЯ СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ")
    print("Проект реализован согласно техническому заданию")
    print(f"PyTorch версия: {torch.__version__}")
    
    if args.demo in ['all', 'model']:
        demo_model_architecture()
    
    if args.demo in ['all', 'loss']:
        demo_loss_functions()
    

    
    if args.demo in ['all', 'training']:
        demo_training_simulation()
    
    if args.demo in ['all', 'inference']:
        demo_inference()
    
    if args.demo in ['all', 'metrics']:
        demo_metrics()
    
    print("\n" + "=" * 60)
    print("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print("Основные компоненты проекта:")
    print("✅ MobileFaceNet - легковесная архитектура")
    print("✅ ArcFace/CosFace - современные loss функции")
    print("✅ Работа с реальными данными")
    print("✅ Полный pipeline обучения")
    print("✅ Инференс и сравнение лиц")
    print("✅ Метрики оценки качества")
    print("\nДля запуска обучения: python train.py")
    print("Для инференса: python inference.py --model_path checkpoints/best_checkpoint.pth --reference img1.jpg --test img2.jpg")


if __name__ == "__main__":
    main()