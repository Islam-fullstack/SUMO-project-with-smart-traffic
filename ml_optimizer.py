#!/usr/bin/env python3
"""
ml_optimizer.py

Модуль оптимизации длительности зеленой фазы с использованием методов машинного обучения.
В этой версии для предсказания оптимальной длительности зеленой фазы применяется алгоритм 
GradientBoostingRegressor из библиотеки scikit-learn. Модель обучается на синтетических данных, 
сгенерированных с помощью случайной выборки по диапазону значений: количество транспортных средств 
и среднее время ожидания.

Для повышения точности используются дополнительные признаки:
  - Произведение (количество транспортных средств × среднее время ожидания);
  - Квадраты каждого входного признака;
  - Сумма исходных признаков.

После обучения модель предсказывает оптимальную длительность зеленой фазы, которая затем ограничивается
минимальным и максимальным значениями.

В конце работы отображается график, показывающий зависимость оптимальной длительности зеленой фазы от количества
транспортных средств для нескольких фиксированных уровней среднего времени ожидания.

Зависимости:
  pip install scikit-learn numpy matplotlib joblib

Пример запуска:
  python ml_optimizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import joblib

class MLTrafficOptimizer:
    def __init__(self, min_duration=10, max_duration=60):
        """
        Инициализация оптимизатора.
        
        Аргументы:
          min_duration (float): минимальное значение длительности зеленой фазы (сек).
          max_duration (float): максимальное значение длительности зеленой фазы (сек).
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        # Используем GradientBoostingRegressor с усилением точности за счет ансамблевого метода.
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self._train_model()

    def _train_model(self):
        """
        Обучает модель на синтетических данных.
        
        Генерация обучающего набора:
          - Количество транспортных средств: случайное значение от 10 до 150.
          - Среднее время ожидания: случайное значение от 0.5 до 20 секунд.
          - Дополнительные признаки: произведение, квадраты входных признаков и сумма.
        
        Целевая переменная рассчитывается по формуле:
          T_eff = min_duration + (max_duration - min_duration) * (vehicle_count/150) + 0.7 * avg_waiting_time
          с последующим ограничением в диапазоне [min_duration, max_duration].
        """
        np.random.seed(42)
        n_samples = 2000  # Увеличиваем количество примеров для повышения точности
        vehicle_counts = np.random.uniform(10, 150, n_samples)
        avg_waiting_times = np.random.uniform(0.5, 20.0, n_samples)
        
        # Дополнительные признаки
        prod_feature = vehicle_counts * avg_waiting_times
        sq_vehicle = vehicle_counts ** 2
        sq_wait = avg_waiting_times ** 2
        sum_features = vehicle_counts + avg_waiting_times
        
        # Формирование обучающего набора признаков
        X_train = np.column_stack((vehicle_counts, avg_waiting_times, prod_feature, sq_vehicle, sq_wait, sum_features))
        
        # Целевая переменная:
        y_train = self.min_duration + (self.max_duration - self.min_duration) * (vehicle_counts / 150.0) + 0.7 * avg_waiting_times
        y_train = np.clip(y_train, self.min_duration, self.max_duration)
        
        self.model.fit(X_train, y_train)
        # Сохраняем обученную модель для дальнейшего использования
        joblib.dump(self.model, "ml_traffic_optimizer_model.pkl")
        print("[INFO] Модель GradientBoostingRegressor обучена на {} примерах.".format(n_samples))

    def predict_duration(self, vehicle_count, avg_waiting_time):
        """
        Предсказывает оптимальную длительность зеленой фазы.

        Аргументы:
          vehicle_count (float): количество транспортных средств.
          avg_waiting_time (float): среднее время ожидания (сек).

        Возвращает:
          float: предсказанная длительность зеленой фазы, ограниченная в диапазоне [min_duration, max_duration].
        """
        prod = vehicle_count * avg_waiting_time
        sq_vehicle = vehicle_count ** 2
        sq_wait = avg_waiting_time ** 2
        sum_features = vehicle_count + avg_waiting_time
        X_input = np.array([[vehicle_count, avg_waiting_time, prod, sq_vehicle, sq_wait, sum_features]])
        predicted = self.model.predict(X_input)[0]
        predicted = max(self.min_duration, min(predicted, self.max_duration))
        return predicted

def plot_prediction_curve(optimizer):
    """
    Строит график зависимости оптимальной длительности зеленой фазы от количества транспортных средств
    для нескольких фиксированных значений среднего времени ожидания.
    
    На графике отображаются кривые для различных уровней среднего времени ожидания, например, 3, 7 и 12 секунд.
    """
    vehicle_range = np.linspace(10, 150, 100)
    waiting_times = [3, 7, 12]
    
    plt.figure(figsize=(10, 6))
    
    for wait in waiting_times:
        predicted_durations = []
        for count in vehicle_range:
            duration = optimizer.predict_duration(count, wait)
            predicted_durations.append(duration)
        plt.plot(vehicle_range, predicted_durations, label=f"Среднее время ожидания = {wait} сек")
    
    plt.xlabel("Количество транспортных средств")
    plt.ylabel("Предсказанная длительность зеленой фазы (сек)")
    plt.title("Зависимость оптимальной длительности зеленой фазы от количества транспорта")
    plt.legend()
    plt.grid(True)
    plt.savefig("ml_optimizer_prediction_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    optimizer = MLTrafficOptimizer(min_duration=10, max_duration=60)
    # Пример предсказания: 50 транспортных средств, 5 секунд ожидания
    example_duration = optimizer.predict_duration(50, 5)
    print(f"Пример предсказанной оптимальной длительности зеленой фазы (50 транспортных средств, 5 сек ожидания): {example_duration:.2f} сек")
    plot_prediction_curve(optimizer)
