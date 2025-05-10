#!/usr/bin/env python3
"""
smart_traffic_light.py

Реализация умного алгоритма управления светофорами с адаптивной регулировкой
продолжительности зеленой фазы. Для оптимизации используется машинное обучение
(модуль ml_optimizer.py с моделью линейной регрессии).

Зависимости:
  pip install numpy matplotlib scikit-learn

Пример запуска:
  python smart_traffic_light.py
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ml_optimizer import MLTrafficOptimizer

class TrafficDensitySimulator:
    """
    Симуляция данных о транспортном потоке.
    Для первых 300 секунд направления N и S имеют высокий трафик, затем — E и W.
    Также возвращается среднее время ожидания для каждого направления.
    """
    def get_vehicle_counts(self, t):
        if t < 300:
            return {"N": 50, "S": 50, "E": 10, "W": 10}
        else:
            return {"N": 10, "S": 10, "E": 50, "W": 50}

    def get_avg_waiting_time(self, t):
        if t < 300:
            return {"N": 5, "S": 5, "E": 2, "W": 2}
        else:
            return {"N": 2, "S": 2, "E": 5, "W": 5}

class AdaptivePhase:
    def __init__(self, phase_id, green_directions, min_duration, max_duration, yellow_time):
        self.phase_id = phase_id
        self.green_directions = green_directions
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.yellow_time = yellow_time
        self.effective_duration = min_duration

    def get_state(self, elapsed_time):
        state = {}
        for d in ["N", "S", "E", "W"]:
            if d in self.green_directions:
                state[d] = "yellow" if elapsed_time >= self.effective_duration - self.yellow_time else "green"
            else:
                state[d] = "red"
        return state

class SmartTrafficLightController:
    def __init__(self, phases, detector, min_cycle_time, max_cycle_time):
        self.phases = phases
        self.detector = detector    # TrafficDensitySimulator
        self.min_cycle_time = min_cycle_time
        self.max_cycle_time = max_cycle_time
        self.current_phase_index = 0
        self.current_phase_elapsed = 0.0
        self.ml_optimizer = MLTrafficOptimizer(min_duration=phases[0].min_duration, 
                                                max_duration=phases[0].max_duration)

    def update(self, dt, current_time):
        counts = self.detector.get_vehicle_counts(current_time)
        avg_waiting = self.detector.get_avg_waiting_time(current_time)
        current_phase = self.phases[self.current_phase_index]
        
        # Рассчитаем суммарное число транспортных средств и среднее время ожидания для зеленых направлений текущей фазы.
        veh_sum = sum([counts.get(d, 0) for d in current_phase.green_directions])
        waiting_values = [avg_waiting.get(d, 0) for d in current_phase.green_directions]
        avg_wait = np.mean(waiting_values) if waiting_values else 0

        # Предсказание оптимальной длительности зеленой фазы с использованием ML модели.
        optimal_duration = self.ml_optimizer.predict_duration(veh_sum, avg_wait)
        current_phase.effective_duration = optimal_duration
        
        # Обновляем время, прошедшее в текущей фазе. Если превышено, переключаем фазу.
        self.current_phase_elapsed += dt
        if self.current_phase_elapsed >= current_phase.effective_duration:
            self.current_phase_index = (self.current_phase_index + 1) % len(self.phases)
            self.current_phase_elapsed = 0.0

    def get_current_phase(self):
        phase = self.phases[self.current_phase_index]
        return phase, self.current_phase_elapsed

def main():
    parser = argparse.ArgumentParser(description="Умный контроллер светофоров с ML-оптимизацией")
    args = parser.parse_args()
    
    # Определяем две фазы: фаза 1 – зеленый для N и S; фаза 2 – зеленый для E и W.
    phase1 = AdaptivePhase("Phase1", ["N", "S"], 10, 60, yellow_time=3)
    phase2 = AdaptivePhase("Phase2", ["E", "W"], 10, 60, yellow_time=3)
    detector = TrafficDensitySimulator()
    controller = SmartTrafficLightController([phase1, phase2], detector, min_cycle_time=20, max_cycle_time=120)
    
    dt = 1.0
    T = 600  # симулируем 600 секунд
    times = []
    effective_durations = []
    
    for t in range(0, T):
        controller.update(dt, t)
        phase, elapsed = controller.get_current_phase()
        times.append(t)
        effective_durations.append(phase.effective_duration)
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, effective_durations, label="Effective Phase Duration", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Duration (s)")
    plt.title("Адаптивная длительность зеленой фазы (умный светофор) с использованием ML")
    plt.legend()
    plt.savefig("smart_traffic_phase_duration.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
