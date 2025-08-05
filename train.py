import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import time
from datetime import datetime

from models.mobilefacenet import MobileFaceNet
from models.losses import ArcFace, CosFace, FocalLoss
from utils.dataset_generator import create_dataloader


class FaceRecognitionTrainer:
    """Класс для обучения модели распознавания лиц"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Создание модели
        self.model = MobileFaceNet(
            embedding_size=config.embedding_size,
            input_size=config.input_size
        ).to(self.device)
        
        # Создание loss функции
        if config.loss_type == 'arcface':
            self.loss_fn = ArcFace(
                embedding_size=config.embedding_size,
                num_classes=config.num_identities,
                margin=config.margin,
                scale=config.scale
            ).to(self.device)
        elif config.loss_type == 'cosface':
            self.loss_fn = CosFace(
                embedding_size=config.embedding_size,
                num_classes=config.num_identities,
                margin=config.margin,
                scale=config.scale
            ).to(self.device)
        else:
            raise ValueError(f"Неподдерживаемый тип loss: {config.loss_type}")
        
        # Оптимизатор
        self.optimizer = optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.loss_fn.parameters()}
        ], lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Планировщик обучения
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
        )
        
        # Для логирования
        self.writer = SummaryWriter(f'runs/face_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Метрики
        self.best_accuracy = 0.0
        self.global_step = 0
        
    def train_epoch(self, dataloader, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        self.loss_fn.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Обнуление градиентов
            self.optimizer.zero_grad()
            
            # Получение эмбеддингов
            embeddings = self.model(images)
            
            # Вычисление loss
            logits = self.loss_fn(embeddings, labels)
            loss = F.cross_entropy(logits, labels)
            
            # Обратное распространение
            loss.backward()
            
            # Градиентный клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.loss_fn.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Вычисление точности
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            total_loss += loss.item()
            
            # Обновление прогресс-бара
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Логирование
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy', correct_predictions/total_samples, self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', current_lr, self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Валидация модели"""
        self.model.eval()
        self.loss_fn.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                embeddings = self.model(images)
                logits = self.loss_fn(embeddings, labels)
                loss = F.cross_entropy(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        # Сохранение последнего чекпоинта
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'last_checkpoint.pth'))
        
        # Сохранение лучшего чекпоинта
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth'))
            print(f"Новый лучший результат! Accuracy: {accuracy:.4f}")
    
    def train(self, train_dataloader, val_dataloader=None):
        """Основной цикл обучения"""
        print(f"Начало обучения на {self.config.epochs} эпох...")
        print(f"Количество параметров модели: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch)
            
            # Валидация
            if val_dataloader is not None:
                val_loss, val_acc = self.validate(val_dataloader)
                
                # Логирование валидации
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                
                # Проверка на лучший результат
                is_best = val_acc > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_acc
                
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"Время эпохи: {time.time() - start_time:.2f}s")
                print("-" * 50)
                
                self.save_checkpoint(epoch, val_acc, is_best)
            else:
                # Если нет валидации, сохраняем по точности обучения
                is_best = train_acc > self.best_accuracy
                if is_best:
                    self.best_accuracy = train_acc
                
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Время эпохи: {time.time() - start_time:.2f}s")
                print("-" * 50)
                
                self.save_checkpoint(epoch, train_acc, is_best)
            
            # Обновление планировщика
            self.scheduler.step()
        
        print(f"Обучение завершено! Лучшая точность: {self.best_accuracy:.4f}")
        self.writer.close()


class Config:
    """Конфигурация для обучения"""
    def __init__(self):
        # Параметры модели
        self.embedding_size = 128
        self.input_size = 112
        
        # Параметры датасета
        self.data_dir = "data/synthetic_faces"
        self.num_identities = 1000
        self.samples_per_identity = 50
        
        # Параметры обучения
        self.batch_size = 64
        self.epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 5e-4
        
        # Параметры loss функции
        self.loss_type = 'arcface'  # 'arcface' или 'cosface'
        self.margin = 0.5
        self.scale = 64.0
        
        # Прочие параметры
        self.num_workers = 4
        self.checkpoint_dir = "checkpoints"
        self.val_split = 0.1


def main():
    parser = argparse.ArgumentParser(description='Обучение модели распознавания лиц')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_faces', 
                       help='Путь к директории с данными')
    parser.add_argument('--batch_size', type=int, default=64, help='Размер батча')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-3, help='Скорость обучения')
    parser.add_argument('--num_identities', type=int, default=1000, help='Количество личностей')
    parser.add_argument('--embedding_size', type=int, default=128, help='Размер эмбеддинга')
    parser.add_argument('--loss_type', type=str, default='arcface', 
                       choices=['arcface', 'cosface'], help='Тип loss функции')
    
    args = parser.parse_args()
    
    # Создание конфигурации
    config = Config()
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.num_identities = args.num_identities
    config.embedding_size = args.embedding_size
    config.loss_type = args.loss_type
    
    # Создание директории для чекпоинтов
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Создание датасетов
    print("Создание датасетов...")
    train_dataloader = create_dataloader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_identities=config.num_identities,
        samples_per_identity=config.samples_per_identity,
        num_workers=config.num_workers,
        shuffle=True
    )
    
    # Создание валидационного датасета (подмножество обучающего)
    val_dataloader = create_dataloader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_identities=min(100, config.num_identities),  # Меньше личностей для валидации
        samples_per_identity=10,
        num_workers=config.num_workers,
        shuffle=False
    )
    
    # Создание тренера и запуск обучения
    trainer = FaceRecognitionTrainer(config)
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()