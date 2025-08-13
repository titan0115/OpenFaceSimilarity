#!/usr/bin/env python3
"""
Скрипт для предобработки данных из data/org в data/pre
"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import urllib.request


def download_face_cascade():
    """Скачивание Haar cascade для детекции лиц"""
    cascade_path = "haarcascade_frontalface_default.xml"
    
    if not os.path.exists(cascade_path):
        print("Скачивание модели для детекции лиц...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)
        print("Модель загружена")
    
    return cascade_path


def detect_and_crop_face(image, face_cascade, output_size=112):
    """
    Детекция и кроп лица из изображения
    """
    # Конвертируем в градации серого для детекции
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Детектируем лица
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Берем самое большое лицо
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    # Добавляем отступы для лучшего кропа
    padding = int(max(w, h) * 0.3)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    # Кропаем лицо
    face_crop = image[y1:y2, x1:x2]
    
    # Изменяем размер до нужного
    face_resized = cv2.resize(face_crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
    
    return face_resized


def preprocess_image(image_path, face_cascade, output_size=112):
    """
    Предобработка одного изображения с детекцией лица
    """
    try:
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Детекция и кроп лица
        face_image = detect_and_crop_face(image, face_cascade, output_size)
        
        if face_image is None:
            # Если лицо не найдено, используем простое изменение размера
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Вычисляем новые размеры
            if h > w:
                new_h = output_size
                new_w = int(w * output_size / h)
            else:
                new_w = output_size
                new_h = int(h * output_size / w)
            
            # Изменяем размер
            image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Создаем квадратное изображение с паддингом
            if new_h < output_size or new_w < output_size:
                padded_image = np.zeros((output_size, output_size, 3), dtype=np.uint8)
                y_offset = (output_size - new_h) // 2
                x_offset = (output_size - new_w) // 2
                padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized
                return cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)
            else:
                return cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        
        return face_image
        
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return None


def preprocess_dataset(input_dir, output_dir, output_size=112, max_images=None):
    """
    Предобработка всего датасета с детекцией лиц
    """
    print(f"Предобработка данных из {input_dir} в {output_dir}")
    print("🔍 Включена детекция лиц - будут извлекаться только области лиц")
    
    # Загружаем модель детекции лиц
    cascade_path = download_face_cascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("❌ Ошибка загрузки модели детекции лиц")
        return
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(image_extensions):
            image_files.append(file)
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Ограничиваем количество если нужно
    if max_images and len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"Ограничено до {max_images} изображений")
    
    processed_count = 0
    faces_detected = 0
    no_face_count = 0
    
    # Обрабатываем каждое изображение
    for i, filename in enumerate(tqdm(image_files, desc="Детекция и кроп лиц")):
        input_path = os.path.join(input_dir, filename)
        
        # Предобработка с детекцией лица
        processed_image = preprocess_image(input_path, face_cascade, output_size)
        
        if processed_image is not None:
            # Проверяем, было ли найдено лицо
            temp_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            temp_faces = face_cascade.detectMultiScale(temp_gray, scaleFactor=1.1, minNeighbors=3)
            
            if len(temp_faces) > 0:
                faces_detected += 1
            else:
                no_face_count += 1
            
            # Сохраняем как PNG с новым именем
            output_filename = f"face_{i:06d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, processed_image)
            processed_count += 1
    
    print(f"\n📊 Статистика обработки:")
    print(f"✅ Успешно обработано: {processed_count} изображений")
    print(f"👤 Лица найдены в: {faces_detected} изображениях")
    print(f"❓ Лица не найдены в: {no_face_count} изображениях")
    print(f"📁 Результаты сохранены в: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Предобработка данных для обучения')
    parser.add_argument('--input_dir', type=str, default='data/org', 
                       help='Путь к исходным данным')
    parser.add_argument('--output_dir', type=str, default='data/pre', 
                       help='Путь для сохранения предобработанных данных')
    parser.add_argument('--output_size', type=int, default=112, 
                       help='Размер выходных изображений')
    parser.add_argument('--max_images', type=int, default=None, 
                       help='Максимальное количество изображений для обработки')
    
    args = parser.parse_args()
    
    # Проверяем существование входной директории
    if not os.path.exists(args.input_dir):
        print(f"❌ Директория {args.input_dir} не найдена!")
        return
    
    # Запускаем предобработку
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_size=args.output_size,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
