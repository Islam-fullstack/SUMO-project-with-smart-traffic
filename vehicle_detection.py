#!/usr/bin/env python3
"""
vehicle_detection.py

Доработанный скрипт для обнаружения и подсчёта транспортных средств с видеопотока.
При запуске программа сначала запрашивает у пользователя выделение области интереса (ROI)
для направлений: North, South, East, West (с помощью cv2.selectROI). Затем для каждого кадра
происходит обнаружение объектов с использованием модели YOLOv8 (ultralytics) и подсчёт числа
транспортных средств в каждой ROI. Счётчики выводятся в левом верхнем углу кадра.

Зависимости:
  pip install opencv-python numpy ultralytics

Пример запуска:
  python vehicle_detection.py --source 1.mp4 --model yolov8n.pt --confidence 0.5
"""

import cv2
import numpy as np
import argparse
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path, confidence_threshold=0.5, classes=None, roi_regions=None):
        """
        Инициализирует детектор транспортных средств.
        
        Аргументы:
          model_path (str): путь к файлу модели YOLOv8 (например, yolov8n.pt)
          confidence_threshold (float): порог уверенности для обнаружения
          classes (list, optional): список классов для обнаружения (по умолчанию ['car', 'truck', 'bus', 'motorcycle'])
          roi_regions (dict, optional): словарь с координатами регионов интереса для каждого направления,
                                        где ключи: "N", "S", "E", "W" и значения: (x1, y1, x2, y2)
        """
        if classes is None:
            classes = ['car', 'truck', 'bus', 'motorcycle']
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes
        self.model = None
        self.roi_regions = roi_regions if roi_regions is not None else {}

    def load_model(self):
        print(f"[INFO] Загрузка модели из {self.model_path}...")
        self.model = YOLO(self.model_path)
        print("[INFO] Модель успешно загружена.")

    def process_frame(self, frame):
        """
        Обрабатывает кадр:
          - Выполняется обнаружение объектов через YOLO
          - Фильтруются объекты по классам и порогу уверенности
          - Определяются центры боксов, и если они попадают в заранее выбранный ROI,
            то увеличивается счетчик для соответствующего направления.
          - На кадре рисуются боксы, подписи и выводятся счетчики.
        
        Возвращает обработанный кадр и список обнаруженных объектов.
        """
        processed_frame = frame.copy()
        results = self.model(frame)[0]
        detection_list = []
        counts = {"N": 0, "S": 0, "E": 0, "W": 0}
        for box in results.boxes:
            conf = float(box.conf)
            cls_idx = int(box.cls)
            label = self.model.names[cls_idx]
            if label in self.classes and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # Определяем, к какому ROI принадлежит центр
                for direction, roi in self.roi_regions.items():
                    rx1, ry1, rx2, ry2 = roi
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        counts[direction] += 1
                        break
                detection_list.append({
                    'label': label,
                    'confidence': conf,
                    'box': (x1, y1, x2, y2)
                })
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(processed_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Выводим результаты в левом верхнем углу
        y0, dy = 30, 30
        for i, direction in enumerate(["N", "S", "E", "W"]):
            text = f"{direction}: {counts[direction]}"
            y = y0 + i * dy
            cv2.putText(processed_frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return processed_frame, detection_list, counts

def select_roi_from_frame(frame, direction):
    """
    Отображает кадр и позволяет пользователю выбрать ROI с помощью cv2.selectROI.
    
    Аргументы:
      frame (ndarray): кадр видео
      direction (str): имя направления ("N", "S", "E", "W")
    
    Возвращает:
      tuple: координаты ROI: (x, y, w, h) --> преобразуются в (x, y, x+w, y+h)
    """
    cv2.namedWindow(f"Выберите ROI для направления {direction}", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(f"Выберите ROI для направления {direction}", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(f"Выберите ROI для направления {direction}")
    x, y, w, h = roi
    return (int(x), int(y), int(x + w), int(y + h))

def main():
    parser = argparse.ArgumentParser(description="Обнаружение транспорта с выделением ROI для каждого направления")
    parser.add_argument("--source", type=str, default="0", help="Путь к видеофайлу или индекс камеры")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Путь к модели YOLO")
    parser.add_argument("--confidence", type=float, default=0.5, help="Порог уверенности")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть видеоисточник.")
        return

    # Считываем первый кадр для калибровки ROI.
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Не удалось получить кадр из источника.")
        cap.release()
        return

    print("[INFO] Выделите области интереса для направлений:")
    directions = ["N", "S", "E", "W"]
    roi_regions = {}
    for d in directions:
        print(f"[INFO] Выделите ROI для направления {d}")
        roi_regions[d] = select_roi_from_frame(frame, d)

    print("[INFO] ROI для направлений:", roi_regions)

    detector = VehicleDetector(model_path=args.model, confidence_threshold=args.confidence, roi_regions=roi_regions)
    detector.load_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output_frame, detections, counts = detector.process_frame(frame)
        cv2.imshow("Vehicle Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
