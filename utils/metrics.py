import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def compute_accuracy(embeddings1, embeddings2, labels, threshold=0.6):
    """
    Вычисление точности верификации лиц
    
    Args:
        embeddings1, embeddings2: эмбеддинги пар изображений
        labels: метки (1 - та же личность, 0 - разные)
        threshold: порог для принятия решения
    
    Returns:
        float: точность
    """
    similarities = F.cosine_similarity(embeddings1, embeddings2)
    predictions = (similarities > threshold).float()
    accuracy = (predictions == labels).float().mean()
    return accuracy.item()


def compute_eer(embeddings1, embeddings2, labels):
    """
    Вычисление Equal Error Rate (EER)
    
    Args:
        embeddings1, embeddings2: эмбеддинги пар изображений
        labels: метки (1 - та же личность, 0 - разные)
    
    Returns:
        float: EER значение
        float: порог при котором достигается EER
    """
    similarities = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Вычисление ROC кривой
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    
    # Поиск точки где FPR = 1 - TPR (EER)
    eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - (1 - tpr))))]
    eer = fpr[np.nanargmin(np.absolute((fpr - (1 - tpr))))]
    
    return eer, eer_threshold


def compute_verification_metrics(embeddings1, embeddings2, labels, thresholds=None):
    """
    Вычисление метрик верификации для разных порогов
    
    Args:
        embeddings1, embeddings2: эмбеддинги пар изображений
        labels: метки (1 - та же личность, 0 - разные)
        thresholds: список порогов для тестирования
    
    Returns:
        dict: словарь с метриками
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    similarities = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
    labels = labels.cpu().numpy()
    
    metrics = {
        'thresholds': thresholds,
        'accuracies': [],
        'true_positive_rates': [],
        'false_positive_rates': [],
        'precisions': [],
        'recalls': []
    }
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        
        # True/False Positives/Negatives
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        # Метрики
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics['accuracies'].append(accuracy)
        metrics['true_positive_rates'].append(tpr)
        metrics['false_positive_rates'].append(fpr)
        metrics['precisions'].append(precision)
        metrics['recalls'].append(tpr)
    
    # EER
    eer, eer_threshold = compute_eer(
        torch.tensor(embeddings1), torch.tensor(embeddings2), torch.tensor(labels)
    )
    metrics['eer'] = eer
    metrics['eer_threshold'] = eer_threshold
    
    # ROC AUC
    fpr_roc, tpr_roc, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr_roc, tpr_roc)
    metrics['roc_auc'] = roc_auc
    
    return metrics


def plot_roc_curve(embeddings1, embeddings2, labels, save_path=None):
    """
    Построение ROC кривой
    
    Args:
        embeddings1, embeddings2: эмбеддинги пар изображений
        labels: метки (1 - та же личность, 0 - разные)
        save_path: путь для сохранения графика
    
    Returns:
        float: AUC значение
    """
    similarities = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
    labels = labels.cpu().numpy()
    
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC кривая (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC кривая для верификации лиц')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC кривая сохранена: {save_path}")
    
    plt.show()
    
    return roc_auc


def plot_verification_metrics(metrics, save_path=None):
    """
    Построение графиков метрик верификации
    
    Args:
        metrics: словарь с метриками (результат compute_verification_metrics)
        save_path: путь для сохранения графика
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    thresholds = metrics['thresholds']
    
    # Точность
    ax1.plot(thresholds, metrics['accuracies'], 'b-', linewidth=2)
    ax1.set_xlabel('Порог')
    ax1.set_ylabel('Точность')
    ax1.set_title('Зависимость точности от порога')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # TPR и FPR
    ax2.plot(thresholds, metrics['true_positive_rates'], 'g-', linewidth=2, label='TPR (Recall)')
    ax2.plot(thresholds, metrics['false_positive_rates'], 'r-', linewidth=2, label='FPR')
    ax2.set_xlabel('Порог')
    ax2.set_ylabel('Скорость')
    ax2.set_title('TPR и FPR в зависимости от порога')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Precision-Recall
    ax3.plot(metrics['recalls'], metrics['precisions'], 'purple', linewidth=2)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall кривая')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # ROC кривая (приближенная)
    ax4.plot(metrics['false_positive_rates'], metrics['true_positive_rates'], 'orange', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title(f'ROC кривая (AUC ≈ {metrics["roc_auc"]:.3f})')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Графики метрик сохранены: {save_path}")
    
    plt.show()


def print_metrics_summary(metrics):
    """
    Вывод сводки метрик
    
    Args:
        metrics: словарь с метриками
    """
    best_acc_idx = np.argmax(metrics['accuracies'])
    best_threshold = metrics['thresholds'][best_acc_idx]
    best_accuracy = metrics['accuracies'][best_acc_idx]
    
    print("=" * 50)
    print("СВОДКА МЕТРИК ВЕРИФИКАЦИИ")
    print("=" * 50)
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"EER: {metrics['eer']:.4f} (порог: {metrics['eer_threshold']:.4f})")
    print(f"Лучшая точность: {best_accuracy:.4f} (порог: {best_threshold:.4f})")
    print(f"TPR при лучшем пороге: {metrics['true_positive_rates'][best_acc_idx]:.4f}")
    print(f"FPR при лучшем пороге: {metrics['false_positive_rates'][best_acc_idx]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    # Тестирование метрик
    batch_size = 100
    embedding_size = 128
    
    # Создание тестовых данных
    embeddings1 = F.normalize(torch.randn(batch_size, embedding_size), p=2, dim=1)
    embeddings2 = F.normalize(torch.randn(batch_size, embedding_size), p=2, dim=1)
    
    # Создание меток (50% совпадений)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Для совпадающих пар делаем эмбеддинги более похожими
    for i in range(batch_size):
        if labels[i] == 1:
            # Добавляем шум к первому эмбеддингу для создания похожего
            noise = torch.randn_like(embeddings1[i]) * 0.1
            embeddings2[i] = F.normalize(embeddings1[i] + noise, p=2, dim=0)
    
    # Вычисление метрик
    print("Вычисление метрик...")
    metrics = compute_verification_metrics(embeddings1, embeddings2, labels)
    
    # Вывод результатов
    print_metrics_summary(metrics)
    
    # Построение графиков
    plot_verification_metrics(metrics)
    plot_roc_curve(embeddings1, embeddings2, labels)