#!/usr/bin/env python3
"""
vehicle_detection.py

Скрипт для обнаружения и подсчета транспортных средств по видеопотоку с камеры наблюдения
с использованием YOLOv8 для обнаружения, OpenCV для обработки видео и NumPy для работы с массивами.

Зависимости:
  - opencv-python
  - numpy
  - ultralytics
  - torch

Установка зависимостей:
  pip install opencv-python numpy ultralytics torch

Пример запуска из командной строки:
  python vehicle_detection.py --source 0 --model yolov8n.pt --confidence 0.5 --output processed_video.mp4

Автор: Islam
Дата: 2025-05-09
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse

class VehicleDetector:
    """
    Класс для обнаружения транспортных средств с использованием модели YOLOv8.
    
    Методы:
      - load_model(): загрузка модели YOLOv8.
      - process_frame(frame): обработка кадра для обнаружения объектов,
          фильтрация по классам и порогу уверенности, отрисовка ограничивающих рамок.
      - count_vehicles_by_direction(frame, detections, regions): подсчет транспортных средств,
          принадлежащих заданным регионам интереса (например, север, юг, восток, запад).
    """
    def __init__(self, model_path, confidence_threshold=0.5, classes=None):
        """
        Инициализация детектора.

        Аргументы:
          model_path (str): Путь к файлу предварительно обученной модели YOLOv8 (например, "yolov8n.pt").
          confidence_threshold (float): Порог уверенности для обнаружения. По умолчанию 0.5.
          classes (list): Список классов для обнаружения. По умолчанию ['car', 'truck', 'bus', 'motorcycle'].
        """
        if classes is None:
            classes = ['car', 'truck', 'bus', 'motorcycle']
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes
        self.model = None

    def load_model(self):
        """
        Загружает модель YOLOv8, используя библиотеку ultralytics.
        """
        print(f"Загрузка модели YOLOv8 из файла: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("Модель успешно загружена.")

    def process_frame(self, frame):
        """
        Обрабатывает один кадр видео:
          - Выполняет обнаружение объектов через YOLOv8.
          - Фильтрует объекты по заданным классам и порогу уверенности.
          - Рисует ограничивающие рамки и подписывает объект (класс и уверенность).

        Аргументы:
          frame (numpy.ndarray): Кадр видеопотока.

        Возвращает:
          processed_frame (numpy.ndarray): Кадр с нанесенными ограничительными рамками.
          detection_list (list): Список обнаруженных объектов с данными: метка, уверенность и координаты (xmin, ymin, xmax, ymax).
        """
        # Копия кадра для отрисовки результатов
        processed_frame = frame.copy()

        # Обнаружение объектов в кадре через модель
        results = self.model(frame)[0]  # YOLOv8 возвращает список результатов для каждого кадра

        detection_list = []
        # Перебор всех обнаруженных боксов
        for box in results.boxes:
            conf = float(box.conf)  # уверенность обнаружения
            cls_idx = int(box.cls)  # индекс класса
            label = self.model.names[cls_idx]  # имя класса

            # Фильтрация по классам и порогу уверенности
            if label in self.classes and conf >= self.confidence_threshold:
                # Получаем координаты ограничивающей рамки
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Добавляем информацию об обнаружении в список
                detection_list.append({
                    'label': label,
                    'confidence': conf,
                    'box': (x1, y1, x2, y2)
                })

                # Рисуем рамку и подпись на кадре
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                text = f"{label}: {conf:.2f}"
                cv2.putText(processed_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return processed_frame, detection_list

    def count_vehicles_by_direction(self, frame, detections, regions):
        """
        Подсчитывает количество транспортных средств в каждом регионе интереса.
        
        Аргументы:
          frame (numpy.ndarray): Исходный кадр (может использоваться для отладки).
          detections (list): Список обнаруженных объектов из метода process_frame.
          regions (dict): Словарь регионов интереса по направлениям.
                          Формат: {'north': (x1, y1, x2, y2), ...}
        
        Возвращает:
          counts (dict): Словарь с количеством транспортных средств по направлениям.
        """
        # Инициализируем словарь счетчиков для каждого направления
        counts = {direction: 0 for direction in regions.keys()}

        # Для каждого обнаруженного объекта определяем координаты центра бокса
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # Проверяем для каждого региона, попадает ли центр бокса в ROI
            for direction, region in regions.items():
                rx1, ry1, rx2, ry2 = region
                if rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2:
                    counts[direction] += 1
        return counts

class VideoProcessor:
    """
    Класс для обработки видеопотока:
      - Открытие видеопотока.
      - Обработка каждого кадра с использованием детектора.
      - Подсчет транспортных средств в регионах интереса.
      - Отображение результата и (при необходимости) сохранение видео.
      - Возможность интерактивной калибровки регионов интереса.
    """
    def __init__(self, source, detector, roi_regions, output_path=None):
        """
        Инициализация процессора видео.

        Аргументы:
          source: Источник видео (путь к файлу или индекс камеры).
          detector: Экземпляр VehicleDetector.
          roi_regions (dict): Словарь с регионами интереса для каждого направления.
          output_path (str, optional): Путь для сохранения обработанного видео.
        """
        self.source = source
        self.detector = detector
        self.roi_regions = roi_regions
        self.output_path = output_path

    def start_processing(self):
        """
        Запускает обработку видео:
          - Открывает видеопоток.
          - Обрабатывает каждый кадр, используя detector.
          - Подсчитывает транспортные средства по регионам интереса.
          - Отображает обработанный кадр с накладываемой информацией.
          - Сохраняет видео, если указан output_path.
        
        Возвращает:
          counts (dict): Словарь с подсчитанными транспортными средствами по направлениям.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть видеопоток.")
            return

        # Получаем параметры видео для VideoWriter
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25

        out = None
        if self.output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка кадра для обнаружения объектов
            processed_frame, detections = self.detector.process_frame(frame)
            # Подсчет транспортных средств по регионам
            counts = self.detector.count_vehicles_by_direction(frame, detections, self.roi_regions)

            # Отображаем информацию о подсчете на кадре
            start_y = 30
            for i, (direction, count) in enumerate(counts.items()):
                text = f"{direction}: {count}"
                cv2.putText(processed_frame, text, (10, start_y + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Рисуем области интереса (ROI) на кадре для наглядности
            for direction, region in self.roi_regions.items():
                rx1, ry1, rx2, ry2 = region
                cv2.rectangle(processed_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                cv2.putText(processed_frame, direction, (rx1, ry1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Vehicle Detection", processed_frame)
            if out is not None:
                out.write(processed_frame)

            # Выход из цикла по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        # Возвращаем последние подсчитанные данные (их можно отправить в модуль управления светофорами)
        return counts

    def calibrate_roi(self):
        """
        Позволяет интерактивно задать регионы интереса для каждого направления.
        Использует cv2.selectROI для выделения областей в одном кадре.

        Возвращает:
          roi_regions (dict): Словарь с региональными координатами в формате (x1, y1, x2, y2).
        """
        print("Начинается калибровка ROI. Для каждого направления выберите область и нажмите Enter.")
        roi_regions = {}
        directions = ['north', 'south', 'east', 'west']
        cap = cv2.VideoCapture(self.source)
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр для калибровки.")
            return roi_regions

        for direction in directions:
            roi = cv2.selectROI(f"Калибровка ROI - {direction}", frame, showCrosshair=True, fromCenter=False)
            x, y, w, h = roi
            roi_regions[direction] = (int(x), int(y), int(x + w), int(y + h))
            cv2.destroyWindow(f"Калибровка ROI - {direction}")

        cap.release()
        self.roi_regions = roi_regions
        print("Калибровка завершена.")
        return roi_regions

def main():
    """
    Главная функция:
      1. Парсит аргументы командной строки.
      2. Инициализирует детектор транспортных средств с моделью YOLOv8.
      3. Задает регионы интереса (можно задать заранее или запустить калибровку).
      4. Создает экземпляр VideoProcessor для обработки видеопотока.
      5. Запускает обработку видео и выводит результаты.
    """
    parser = argparse.ArgumentParser(description="Обнаружение и подсчет транспортных средств с YOLOv8")
    parser.add_argument("--source", type=str, default="0",
                        help="Источник видео (файл или индекс камеры). По умолчанию 0 (веб-камера).")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Путь к файлу модели YOLO. По умолчанию 'yolov8n.pt'.")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Порог уверенности для обнаружения объектов. По умолчанию 0.5.")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения обработанного видео (опционально).")
    args = parser.parse_args()

    # Преобразование источника в индекс камеры, если возможно, иначе используем путь к видеофайлу
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Инициализируем экземпляр VehicleDetector и загружаем модель
    detector = VehicleDetector(model_path=args.model, confidence_threshold=args.confidence)
    detector.load_model()

    # Задаем регионы интереса заранее. Можно заменить на интерактивную калибровку, вызвав метод calibrate_roi().
    roi_regions = {
        "north": (0, 0, 640, 240),
        "south": (0, 240, 640, 480),
        "east":  (320, 0, 640, 480),
        "west":  (0, 0, 320, 480)
    }

    # Пример интерактивной калибровки (раскомментируйте при необходимости):
    # processor_instance = VideoProcessor(source, detector, roi_regions)
    # roi_regions = processor_instance.calibrate_roi()

    # Инициализируем VideoProcessor
    processor = VideoProcessor(source, detector, roi_regions, output_path=args.output)
    traffic_data = processor.start_processing()

    # Вывод результатов подсчета транспортных средств по направлениям
    print("Результаты подсчета транспортных средств по направлениям:")
    for direction, count in traffic_data.items():
        print(f"{direction}: {count}")

if __name__ == '__main__':
    main()
