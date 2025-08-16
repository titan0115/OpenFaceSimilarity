import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from models.model import MobileFaceNet
from models.losses import ArcFace, CosFace, CurricularFace, FocalLoss
from utils.simple_dataset import create_dataloader
from config import Config as ProjectConfig


class FaceRecognitionTrainer:
    """Класс для обучения модели распознавания лиц"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Инициализация AMP
        self.use_amp = config.training.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print(f"✅ AMP включен")
        else:
            self.scaler = None
            print(f"⚠️ AMP отключен (недоступен на {self.device} или отключен в настройках)")
        
        # Создание модели
        self.model = MobileFaceNet(
            embedding_size=config.model.embedding_size,
            input_size=config.model.input_size
        ).to(self.device)
        
        # Определение параметров для loss функции
        embedding_size = config.model.embedding_size
        num_identities = config.data.num_identities
        loss_type = config.training.loss_type
        margin = config.training.margin
        scale = config.training.scale
        
        # Создание loss функции
        if loss_type == 'arcface':
            self.loss_fn = ArcFace(
                embedding_size=embedding_size,
                num_classes=num_identities,
                margin=margin,
                scale=scale
            ).to(self.device)
        elif loss_type == 'cosface':
            self.loss_fn = CosFace(
                embedding_size=embedding_size,
                num_classes=num_identities,
                margin=margin,
                scale=scale
            ).to(self.device)
        elif loss_type == 'curricularface':
            self.loss_fn = CurricularFace(
                embedding_size=embedding_size,
                num_classes=num_identities,
                margin=margin,
                scale=scale,
                alpha=config.training.alpha
            ).to(self.device)
        else:
            raise ValueError(f"Неподдерживаемый тип loss: {loss_type}")
        
        # Определение параметров для оптимизатора
        learning_rate = config.training.learning_rate
        weight_decay = config.training.weight_decay
        epochs = config.training.epochs
        
        # Оптимизатор
        self.optimizer = optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.loss_fn.parameters()}
        ], lr=learning_rate, weight_decay=weight_decay)
        
        # Планировщик обучения
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
        
        # Для логирования
        self.writer = SummaryWriter(f'runs/face_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Метрики
        self.best_accuracy = 0.0
        self.global_step = 0
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка чекпоинта с поддержкой AMP"""
        if os.path.exists(checkpoint_path):
            print(f"Загрузка чекпоинта: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.loss_fn.load_state_dict(checkpoint['loss_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Загрузка состояния AMP scaler если есть
            if self.scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.best_accuracy = checkpoint.get('accuracy', 0.0)
            return checkpoint['epoch']
        else:
            print(f"Чекпоинт не найден: {checkpoint_path}")
            return 0
        
    def train_epoch(self, dataloader, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        self.loss_fn.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        epochs = self.config.training.epochs
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Обнуление градиентов
            self.optimizer.zero_grad()
            
            # Использование AMP для прямого прохода
            if self.use_amp:
                with autocast():
                    embeddings = self.model(images)
                    logits = self.loss_fn(embeddings, labels)
                    loss = F.cross_entropy(logits, labels)
                
                # Масштабированное обратное распространение
                self.scaler.scale(loss).backward()
                
                # Градиентный клиппинг с учетом масштабирования
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.loss_fn.parameters(), max_norm=1.0)
                
                # Обновление параметров
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Обычный прямой проход без AMP
                embeddings = self.model(images)
                logits = self.loss_fn(embeddings, labels)
                loss = F.cross_entropy(logits, labels)
                
                # Обратное распространение
                loss.backward()
                
                # Градиентный клиппинг
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.loss_fn.parameters(), max_norm=1.0)
                
                # Обновление параметров
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
                
                # Использование AMP для валидации (если включен)
                if self.use_amp:
                    with autocast():
                        embeddings = self.model(images)
                        logits = self.loss_fn(embeddings, labels)
                        loss = F.cross_entropy(logits, labels)
                else:
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
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'accuracy': accuracy,
            'config': self.config,
            'use_amp': self.use_amp
        }
        
        # Получение checkpoint_dir из конфигурации
        checkpoint_dir = self.config.training.checkpoint_dir
        
        # Сохранение последнего чекпоинта
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'last_checkpoint.pth'))
        
        # Сохранение лучшего чекпоинта
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_checkpoint.pth'))
            print(f"Новый лучший результат! Accuracy: {accuracy:.4f}")
    
    def train(self, train_dataloader, val_dataloader=None):
        """Основной цикл обучения"""
        # Получение параметров из конфигурации
        epochs = self.config.training.epochs
        save_frequency = self.config.training.save_frequency
        validation_frequency = self.config.training.validation_frequency
        
        print(f"Начало обучения на {epochs} эпох...")
        print(f"Валидация будет проводиться каждые {validation_frequency} эпох")
        print(f"Количество параметров модели: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch)
            
            # Валидация (только в определенные эпохи)
            should_validate = (val_dataloader is not None and 
                             (epoch + 1) % validation_frequency == 0)
            
            if should_validate:
                val_loss, val_acc = self.validate(val_dataloader)
                
                # Логирование валидации
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                
                # Проверка на лучший результат
                is_best = val_acc > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_acc
                
                print(f"Epoch {epoch+1}/{epochs}")
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
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Валидация пропущена (следующая через {validation_frequency - ((epoch + 1) % validation_frequency)} эпох)")
                print(f"Время эпохи: {time.time() - start_time:.2f}s")
                print("-" * 50)
                
                # Сохраняем чекпоинт только если это лучший результат или по частоте сохранения
                if is_best or (epoch + 1) % save_frequency == 0:
                    self.save_checkpoint(epoch, train_acc, is_best)
            
            # Обновление планировщика
            self.scheduler.step()
        
        print(f"Обучение завершено! Лучшая точность: {self.best_accuracy:.4f}")
        self.writer.close()





def main():
    # Использование конфигурации по умолчанию
    config = ProjectConfig()
    print("Используется конфигурация из config.py")
    
    # Создание необходимых директорий
    config.create_directories()
    
    # Создание датасетов из предобработанных данных
    print("Создание датасетов из предобработанных данных...")
    
    # Автоматическое определение количества изображений
    import glob
    image_files = glob.glob(os.path.join(config.data.data_dir, "*.png")) + \
                 glob.glob(os.path.join(config.data.data_dir, "*.jpg")) + \
                 glob.glob(os.path.join(config.data.data_dir, "*.jpeg"))
    
    if len(image_files) == 0:
        print(f"❌ В папке {config.data.data_dir} нет изображений!")
        print(f"Поместите предобработанные изображения в папку {config.data.data_dir}")
        return
    
    actual_num_identities = min(len(image_files), config.data.num_identities)
    config.data.num_identities = actual_num_identities
    print(f"Найдено {len(image_files)} изображений, используем {actual_num_identities}")
    
    # Создание train и val датасетов
    train_dataloader, val_dataloader = create_dataloader(
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        input_size=config.model.input_size,
        augment=True,
        max_images=actual_num_identities,
        num_workers=config.data.num_workers,
        shuffle=True,
        val_split=config.data.val_split
    )
    
    # Создание тренера и запуск обучения
    trainer = FaceRecognitionTrainer(config)
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()