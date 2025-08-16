import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFace(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) для распознавания лиц
    Основан на статье: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0):
        super(ArcFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin  # Угловой отступ m
        self.scale = scale    # Масштабирующий фактор s
        
        # Инициализация весов классификатора
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Предвычисленные значения для оптимизации
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Пороговое значение
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: нормализованные эмбеддинги лиц [batch_size, embedding_size]
            labels: метки классов [batch_size]
        Returns:
            logits: выходные логиты для вычисления cross-entropy loss
        """
        # Нормализация весов
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Вычисление косинуса угла между эмбеддингом и весом
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Вычисление cos(θ + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Применение порогового значения для численной стабильности
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Создание one-hot маски для целевых классов
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Применение углового отступа только к целевым классам
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Масштабирование для стабильного обучения
        output *= self.scale
        
        return output


class CurricularFace(nn.Module):
    """
    CurricularFace - адаптивная версия ArcFace с динамическим margin
    Основан на статье: "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition"
    """
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0, alpha=0.01):
        super(CurricularFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin  # Базовый угловой отступ m
        self.scale = scale    # Масштабирующий фактор s
        self.alpha = alpha    # Коэффициент адаптации
        
        # Инициализация весов классификатора
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Предвычисленные значения для оптимизации
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Пороговое значение
        self.mm = math.sin(math.pi - margin) * margin
        
        # Буфер для хранения статистики обучения
        self.register_buffer('t', torch.zeros(1))
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: нормализованные эмбеддинги лиц [batch_size, embedding_size]
            labels: метки классов [batch_size]
        Returns:
            logits: выходные логиты для вычисления cross-entropy loss
        """
        # Нормализация весов
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Вычисление косинуса угла между эмбеддингом и весом
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Вычисление cos(θ + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Применение порогового значения для численной стабильности
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Создание one-hot маски для целевых классов
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # CurricularFace: адаптивное модулирование negative logits
        with torch.no_grad():
            self.t = self.alpha * self.t + (1 - self.alpha) * cosine.mean()
        
        # Применение curricular weighting для negative samples
        negative_cosine = cosine - one_hot * self.margin
        curricular_weight = 1 + torch.relu(negative_cosine.detach() - self.t)
        negative_cosine = negative_cosine * curricular_weight
        
        # Финальный вывод с curricular modulation
        output = one_hot * phi + (1.0 - one_hot) * negative_cosine
        
        # Масштабирование для стабильного обучения
        output *= self.scale
        
        return output


class CosFace(nn.Module):
    """
    CosFace (Large Margin Cosine Loss) для распознавания лиц
    Альтернативная реализация margin-based loss функции
    """
    def __init__(self, embedding_size, num_classes, margin=0.35, scale=64.0):
        super(CosFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # Нормализация весов
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Вычисление косинуса
        cosine = F.linear(embeddings, weight_norm)
        
        # Создание one-hot маски
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Применение косинусного отступа
        output = cosine - one_hot * self.margin
        output *= self.scale
        
        return output


class FocalLoss(nn.Module):
    """
    Focal Loss для балансировки сложных и простых примеров
    Может использоваться в комбинации с ArcFace
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def cosine_similarity_loss(embeddings1, embeddings2, labels):
    """
    Простая функция для вычисления косинусного расстояния между эмбеддингами
    Используется для валидации и инференса
    
    Args:
        embeddings1, embeddings2: нормализованные эмбеддинги
        labels: 1 если та же личность, 0 если разная
    """
    similarity = F.cosine_similarity(embeddings1, embeddings2)
    loss = F.binary_cross_entropy_with_logits(similarity, labels.float())
    return loss


if __name__ == "__main__":
    # Тестирование loss функций
    batch_size = 4
    embedding_size = 128
    num_classes = 1000
    
    # Создание тестовых данных
    embeddings = F.normalize(torch.randn(batch_size, embedding_size), p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Тест ArcFace
    arcface = ArcFace(embedding_size, num_classes)
    output = arcface(embeddings, labels)
    loss = F.cross_entropy(output, labels)
    print(f"ArcFace output shape: {output.shape}")
    print(f"ArcFace loss: {loss.item():.4f}")
    
    # Тест CosFace
    cosface = CosFace(embedding_size, num_classes)
    output = cosface(embeddings, labels)
    loss = F.cross_entropy(output, labels)
    print(f"CosFace output shape: {output.shape}")
    print(f"CosFace loss: {loss.item():.4f}")
    
    # Тест CurricularFace
    curricularface = CurricularFace(embedding_size, num_classes)
    output = curricularface(embeddings, labels)
    loss = F.cross_entropy(output, labels)
    print(f"CurricularFace output shape: {output.shape}")
    print(f"CurricularFace loss: {loss.item():.4f}")