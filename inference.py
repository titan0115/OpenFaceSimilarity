import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import argparse
from torchvision import transforms

from models.mobilefacenet import MobileFaceNet


class FaceComparator:
    """Класс для сравнения лиц и инференса модели"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загрузка модели
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Трансформации для входных изображений
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Модель загружена на устройство: {self.device}")
    
    def _load_model(self, model_path):
        """Загрузка предобученной модели"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Получение конфигурации из чекпоинта
        if 'config' in checkpoint:
            config = checkpoint['config']
            embedding_size = config.embedding_size
            input_size = config.input_size
        else:
            # Значения по умолчанию
            embedding_size = 128
            input_size = 112
            print("Конфигурация не найдена в чекпоинте, используем значения по умолчанию")
        
        # Создание модели
        model = MobileFaceNet(embedding_size=embedding_size, input_size=input_size)
        
        # Загрузка весов
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # Если сохранены только веса модели
        
        model.to(self.device)
        return model
    
    def preprocess_image(self, image_path_or_array):
        """Предобработка изображения для модели"""
        if isinstance(image_path_or_array, str):
            # Загрузка из файла
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path_or_array}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Уже загруженное изображение
            image = image_path_or_array
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Предполагаем, что это BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Преобразование в PIL Image
        image = Image.fromarray(image)
        
        # Применение трансформаций
        image_tensor = self.transform(image)
        
        return image_tensor.unsqueeze(0)  # Добавляем batch dimension
    
    def get_embedding(self, image):
        """Получение эмбеддинга для изображения"""
        with torch.no_grad():
            image_tensor = self.preprocess_image(image).to(self.device)
            embedding = self.model(image_tensor)
            return embedding.cpu().numpy().flatten()
    
    def compare_faces(self, image1, image2, threshold=0.6):
        """
        Сравнение двух лиц
        
        Args:
            image1, image2: пути к изображениям или numpy массивы
            threshold: порог для определения схожести
            
        Returns:
            dict: результат сравнения с метриками
        """
        # Получение эмбеддингов
        embedding1 = self.get_embedding(image1)
        embedding2 = self.get_embedding(image2)
        
        # Вычисление косинусного расстояния
        cosine_similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Вычисление евклидова расстояния
        euclidean_distance = np.linalg.norm(embedding1 - embedding2)
        
        # Определение совпадения
        is_same_person = cosine_similarity > threshold
        
        return {
            'cosine_similarity': float(cosine_similarity),
            'euclidean_distance': float(euclidean_distance),
            'is_same_person': bool(is_same_person),
            'confidence': float(cosine_similarity) if is_same_person else float(1 - cosine_similarity),
            'threshold': threshold
        }
    
    def verify_identity(self, reference_image, test_image, threshold=0.6):
        """
        Верификация личности относительно референсного изображения
        
        Args:
            reference_image: референсное изображение
            test_image: проверяемое изображение
            threshold: порог верификации
            
        Returns:
            bool: True если та же личность, False иначе
        """
        result = self.compare_faces(reference_image, test_image, threshold)
        return result['is_same_person']
    
    def batch_compare(self, reference_image, test_images, threshold=0.6):
        """
        Сравнение одного референсного изображения с множеством тестовых
        
        Args:
            reference_image: референсное изображение
            test_images: список путей к тестовым изображениям
            threshold: порог для определения схожести
            
        Returns:
            list: список результатов для каждого тестового изображения
        """
        reference_embedding = self.get_embedding(reference_image)
        results = []
        
        for test_image in test_images:
            try:
                test_embedding = self.get_embedding(test_image)
                
                cosine_similarity = np.dot(reference_embedding, test_embedding) / (
                    np.linalg.norm(reference_embedding) * np.linalg.norm(test_embedding)
                )
                
                euclidean_distance = np.linalg.norm(reference_embedding - test_embedding)
                is_same_person = cosine_similarity > threshold
                
                results.append({
                    'image_path': test_image,
                    'cosine_similarity': float(cosine_similarity),
                    'euclidean_distance': float(euclidean_distance),
                    'is_same_person': bool(is_same_person),
                    'confidence': float(cosine_similarity) if is_same_person else float(1 - cosine_similarity)
                })
            except Exception as e:
                results.append({
                    'image_path': test_image,
                    'error': str(e),
                    'is_same_person': False,
                    'confidence': 0.0
                })
        
        return results
    
    def find_best_match(self, reference_image, candidate_images):
        """
        Поиск наилучшего совпадения среди кандидатов
        
        Args:
            reference_image: referencer изображение
            candidate_images: список кандидатов
            
        Returns:
            dict: информация о лучшем совпадении
        """
        results = self.batch_compare(reference_image, candidate_images)
        
        # Фильтрация ошибок
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return None
        
        # Поиск максимального косинусного сходства
        best_match = max(valid_results, key=lambda x: x['cosine_similarity'])
        
        return best_match


def main():
    parser = argparse.ArgumentParser(description='Инференс модели распознавания лиц')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Путь к файлу модели (.pth)')
    parser.add_argument('--reference', type=str, required=True,
                       help='Путь к референсному изображению')
    parser.add_argument('--test', type=str, required=True,
                       help='Путь к тестовому изображению или директории с изображениями')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Порог для определения схожести (по умолчанию: 0.6)')
    parser.add_argument('--batch', action='store_true',
                       help='Режим пакетного сравнения (test должен быть директорией)')
    
    args = parser.parse_args()
    
    # Проверка существования файлов
    if not os.path.exists(args.model_path):
        print(f"Ошибка: файл модели не найден: {args.model_path}")
        return
    
    if not os.path.exists(args.reference):
        print(f"Ошибка: референсное изображение не найдено: {args.reference}")
        return
    
    if not os.path.exists(args.test):
        print(f"Ошибка: тестовое изображение/директория не найдены: {args.test}")
        return
    
    # Создание компаратора
    comparator = FaceComparator(args.model_path)
    
    if args.batch and os.path.isdir(args.test):
        # Пакетное сравнение
        test_images = []
        for filename in os.listdir(args.test):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_images.append(os.path.join(args.test, filename))
        
        if not test_images:
            print("В указанной директории не найдены изображения")
            return
        
        print(f"Сравнение с {len(test_images)} изображениями...")
        results = comparator.batch_compare(args.reference, test_images, args.threshold)
        
        # Вывод результатов
        matches = [r for r in results if r.get('is_same_person', False)]
        
        print(f"\nРезультаты сравнения:")
        print(f"Всего изображений: {len(results)}")
        print(f"Совпадений: {len(matches)}")
        print("-" * 80)
        
        for result in sorted(results, key=lambda x: x.get('cosine_similarity', 0), reverse=True):
            if 'error' in result:
                print(f"❌ {result['image_path']}: ОШИБКА - {result['error']}")
            else:
                status = "✅ СОВПАДЕНИЕ" if result['is_same_person'] else "❌ НЕ СОВПАДАЕТ"
                print(f"{status} | {result['image_path']}")
                print(f"   Косинусное сходство: {result['cosine_similarity']:.4f}")
                print(f"   Евклидово расстояние: {result['euclidean_distance']:.4f}")
                print(f"   Уверенность: {result['confidence']:.4f}")
                print()
        
        # Лучшее совпадение
        best_match = comparator.find_best_match(args.reference, test_images)
        if best_match:
            print(f"🏆 ЛУЧШЕЕ СОВПАДЕНИЕ:")
            print(f"   Изображение: {best_match['image_path']}")
            print(f"   Косинусное сходство: {best_match['cosine_similarity']:.4f}")
            print(f"   Та же личность: {'Да' if best_match['is_same_person'] else 'Нет'}")
    
    else:
        # Сравнение одного изображения
        result = comparator.compare_faces(args.reference, args.test, args.threshold)
        
        print(f"\nРезультат сравнения:")
        print(f"Референсное изображение: {args.reference}")
        print(f"Тестовое изображение: {args.test}")
        print("-" * 50)
        print(f"Косинусное сходство: {result['cosine_similarity']:.4f}")
        print(f"Евклидово расстояние: {result['euclidean_distance']:.4f}")
        print(f"Порог: {result['threshold']}")
        print(f"Та же личность: {'Да' if result['is_same_person'] else 'Нет'}")
        print(f"Уверенность: {result['confidence']:.4f}")
        
        if result['is_same_person']:
            print("✅ СОВПАДЕНИЕ: Изображения принадлежат одной личности")
        else:
            print("❌ НЕ СОВПАДАЕТ: Изображения принадлежат разным личностям")


if __name__ == "__main__":
    main()