#!/usr/bin/env python3
"""
smart_traffic_light.py

Скрипт для моделирования умного алгоритма управления светофорами с адаптивными фазами,
учитывающего плотность транспортного потока и длину очередей. При этом адаптивная логика
переключения фаз изменяет продолжительность зеленой фазы в зависимости от приоритетных баллов,
рассчитываемых по количеству транспортных средств и времени их ожидания.

Зависимости:
  - numpy
  - matplotlib
  - pandas
  - scikit-learn (опционально, для более сложных алгоритмов)

Установка зависимостей:
  pip install numpy matplotlib pandas scikit-learn

Пример запуска из командной строки:
  python smart_traffic_light.py --sim_time 3600 --dt 0.1 --min_phase_duration 10 --max_phase_duration 60 --yellow_time 3 --output results.csv

Автор: Islam
Дата: 2025-05-09
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression  # опционально для расширенных алгоритмов

###############################################################################
# Класс Lane
###############################################################################
class Lane:
    """
    Представляет простую модель полосы движения по направлению.

    Атрибуты:
      - direction: направление ("N", "S", "E", "W")
      - length: длина полосы (метры)
      - queue: текущее число транспортных средств в очереди
      - cumulative_waiting_time: накопленное время ожидания транспортных средств
    """
    def __init__(self, direction, length=500):
        self.direction = direction
        self.length = length
        self.queue = 0
        self.cumulative_waiting_time = 0.0

    def update(self, dt, arrivals, traffic_light_state, departure_rate=1.0):
        """
        Обновляет состояние полосы:
          - Добавляет прибытия (arrivals) к очереди.
          - Накопление времени ожидания пропорционально количеству транспортных средств.
          - Если сигнал "green", транспортные средства покидают очередь с заданной интенсивностью.
        
        Аргументы:
          dt: шаг времени симуляции (сек).
          arrivals: число транспортных средств, прибывших за dt.
          traffic_light_state: состояние светофора ("green", "yellow" или "red").
          departure_rate: интенсивность выезда транспортных средств (транспорт/сек).
        """
        self.queue += arrivals
        self.cumulative_waiting_time += self.queue * dt
        if traffic_light_state == "green":
            departures = np.random.poisson(departure_rate * dt)
            departures = min(self.queue, departures)
            self.queue -= departures
        return self.queue

###############################################################################
# Класс TrafficDensityDetector
###############################################################################
class TrafficDensityDetector:
    """
    Детектор плотности транспортного потока собирает данные о движении на полосах.

    Атрибуты:
      - detector_id: уникальный идентификатор детектора.
      - lanes: словарь с объектами Lane для каждого направления.
    
    Методы:
      - get_vehicle_count(direction): возвращает количество транспортных средств (очередь) в заданном направлении.
      - get_queue_lengths(): возвращает длину очереди для каждого направления.
      - get_average_waiting_time(): возвращает среднее время ожидания транспортных средств по направлениям.
      - get_densities(): возвращает плотность потока (отношение числа машин к длине полосы).
      - update(frame=None): обновление данных (в реальном случае – обработка видеокадра, здесь – заглушка).
    """
    def __init__(self, detector_id, lanes):
        self.detector_id = detector_id
        self.lanes = lanes

    def get_vehicle_count(self, direction):
        return self.lanes[direction].queue if direction in self.lanes else 0

    def get_queue_lengths(self):
        return {direction: lane.queue for direction, lane in self.lanes.items()}

    def get_average_waiting_time(self):
        avg_wait = {}
        for direction, lane in self.lanes.items():
            avg_wait[direction] = lane.cumulative_waiting_time / lane.queue if lane.queue > 0 else 0.0
        return avg_wait

    def get_densities(self):
        # Расчет плотности как отношение числа транспортных средств к длине полосы.
        return {direction: lane.queue / lane.length for direction, lane in self.lanes.items()}

    def update(self, frame=None):
        # В реальной системе здесь происходит обработка видеоданных.
        pass

###############################################################################
# Класс AdaptivePhase
###############################################################################
class AdaptivePhase:
    """
    Представляет фазу работы светофора с адаптивными временными ограничениями.

    Атрибуты:
      - phase_id: уникальный идентификатор фазы.
      - green_directions: список направлений с зеленым сигналом.
      - min_duration: минимальное время фазы (сек).
      - max_duration: максимальное время фазы (сек).
      - yellow_time: время желтого сигнала перед сменой фазы.
      - effective_duration: текущая адаптивная длительность фазы (изначально min_duration).
    
    Метод:
      - get_state(elapsed_time): возвращает состояния светофора для всех направлений.
    """
    def __init__(self, phase_id, green_directions, min_duration, max_duration, yellow_time):
        self.phase_id = phase_id
        self.green_directions = green_directions
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.yellow_time = yellow_time
        self.effective_duration = min_duration  # изначально

    def get_state(self, elapsed_time):
        state = {}
        for d in ["N", "S", "E", "W"]:
            if d in self.green_directions:
                if elapsed_time >= self.effective_duration - self.yellow_time:
                    state[d] = "yellow"
                else:
                    state[d] = "green"
            else:
                state[d] = "red"
        return state

###############################################################################
# Класс SmartTrafficLightController
###############################################################################
class SmartTrafficLightController:
    """
    Умный контроллер светофоров, адаптирующий продолжительность зеленых фаз на основе данных детектора.

    Атрибуты:
      - controller_id: уникальный идентификатор.
      - phases: список объектов AdaptivePhase.
      - detector: экземпляр TrafficDensityDetector.
      - min_cycle_time, max_cycle_time: ограничения на весь цикл переключения.

    Методы:
      - update(dt): обновляет внутренний таймер и при необходимости переключает фазу.
      - get_current_phase(), get_current_state(), get_remaining_time(): возвращают информацию о текущей фазе.
      - adapt_phase_duration(): изменяет эффективную длительность текущей фазы на основе данных детектора.
      - calculate_priority_score(direction): вычисляет приоритет для направления (кол-во транспортных средств + время ожидания).
      - decide_next_phase(): выбирает следующую фазу с наибольшим приоритетом.
      - reset(): сбрасывает контроллер в исходное состояние.
    """
    def __init__(self, controller_id, phases, detector, min_cycle_time, max_cycle_time):
        self.controller_id = controller_id
        self.phases = phases
        self.detector = detector
        self.min_cycle_time = min_cycle_time
        self.max_cycle_time = max_cycle_time

        self.current_phase_index = 0
        self.current_phase_elapsed = 0.0
        self.phases[self.current_phase_index].effective_duration = self.phases[self.current_phase_index].min_duration

    def update(self, dt):
        self.current_phase_elapsed += dt
        # Адаптация длительности текущей фазы в зависимости от данных детектора
        self.adapt_phase_duration()

        current_phase = self.phases[self.current_phase_index]
        if self.current_phase_elapsed >= current_phase.effective_duration:
            self.decide_next_phase()
            self.current_phase_elapsed = 0.0

    def get_current_phase(self):
        current_phase = self.phases[self.current_phase_index]
        return current_phase, self.current_phase_elapsed

    def get_current_state(self):
        phase, elapsed = self.get_current_phase()
        return phase.get_state(elapsed)

    def get_remaining_time(self):
        phase, elapsed = self.get_current_phase()
        return phase.effective_duration - elapsed

    def adapt_phase_duration(self):
        """
        Вычисляет совокупный приоритет для направлений в текущей фазе и адаптирует её продолжительность.
        При высоком трафике зеленая фаза удлиняется, при малом – сокращается.
        """
        current_phase = self.phases[self.current_phase_index]
        total_score = 0.0
        for d in current_phase.green_directions:
            score = self.calculate_priority_score(d)
            total_score += score
        # Нормализуем балл: считаем, что максимальный суммарный балл равен 10 (подбор параметра)
        normalized_score = min(total_score / 10.0, 1.0)
        new_duration = current_phase.min_duration + (current_phase.max_duration - current_phase.min_duration) * normalized_score
        current_phase.effective_duration = new_duration

    def calculate_priority_score(self, direction):
        """
        Рассчитывает приоритетный балл для направления как сумму количества транспортных средств
        и взвешенного среднего времени ожидания.
        """
        vehicle_count = self.detector.get_vehicle_count(direction)
        avg_wait = self.detector.get_average_waiting_time().get(direction, 0.0)
        # Вес для времени ожидания можно настроить, здесь 0.5
        score = vehicle_count + 0.5 * avg_wait
        return score

    def decide_next_phase(self):
        """
        Выбирает следующую фазу, суммируя приоритеты для зеленых направлений каждой фазы,
        и переключается на фазу с наибольшим баллом.
        """
        phase_scores = []
        for phase in self.phases:
            score = sum(self.calculate_priority_score(d) for d in phase.green_directions)
            phase_scores.append(score)
        next_phase_index = np.argmax(phase_scores)
        self.current_phase_index = next_phase_index
        self.phases[self.current_phase_index].effective_duration = self.phases[self.current_phase_index].min_duration

    def reset(self):
        self.current_phase_index = 0
        self.current_phase_elapsed = 0.0
        self.phases[self.current_phase_index].effective_duration = self.phases[self.current_phase_index].min_duration

###############################################################################
# Класс IntersectionWithSmartLights
###############################################################################
class IntersectionWithSmartLights:
    """
    Перекресток с умным управлением светофоров, объединяющий контроллер, полосы и детектор.

    Атрибуты:
      - intersection_id: идентификатор перекрестка.
      - controller: экземпляр SmartTrafficLightController.
      - lanes: словарь объектов Lane по направлениям.
      - detector: экземпляр TrafficDensityDetector.
    
    Методы:
      - update(dt, traffic_generator): обновляет состояние перекрестка (прибыли, светофор, данные детектора).
      - get_traffic_light_states(): возвращает текущее состояние светофоров.
      - collect_statistics(): собирает статистику (очереди, время ожидания).
    """
    def __init__(self, intersection_id, controller, lanes, detector):
        self.intersection_id = intersection_id
        self.controller = controller
        self.lanes = lanes
        self.detector = detector

    def update(self, dt, traffic_generator):
        # Обновляем контроллер, который адаптирует фазу на основании данных детектора
        self.controller.update(dt)
        current_state = self.controller.get_current_state()
        # Генерируем новые прибытия для каждой полосы
        arrivals = traffic_generator.generate(dt)
        for direction, lane in self.lanes.items():
            lane.update(dt, arrivals.get(direction, 0), current_state.get(direction, "red"))
        self.detector.update()

    def get_traffic_light_states(self):
        return self.controller.get_current_state()

    def collect_statistics(self):
        stats = {}
        for direction, lane in self.lanes.items():
            stats[direction] = {
                "queue_length": lane.queue,
                "cumulative_waiting_time": lane.cumulative_waiting_time
            }
        return stats

###############################################################################
# Класс TrafficGenerator
###############################################################################
class TrafficGenerator:
    """
    Генерирует прибытия транспортных средств для каждого направления согласно пуассоновскому распределению.

    Атрибут:
      - arrival_rates: словарь интенсивностей прибытия (транспорт/сек).
    """
    def __init__(self, arrival_rates):
        self.arrival_rates = arrival_rates

    def generate(self, dt):
        return {direction: np.random.poisson(rate * dt)
                for direction, rate in self.arrival_rates.items()}

###############################################################################
# Класс SmartTrafficLightSimulation
###############################################################################
class SmartTrafficLightSimulation:
    """
    Моделирует симуляцию умного управления светофорами.

    Атрибуты:
      - intersections: список объектов IntersectionWithSmartLights.
      - simulation_time: общее время симуляции (сек).
      - dt: шаг времени симуляции (сек).
      - traffic_generator: объект для генерации прибытия транспортных средств.
    
    Методы:
      - run(): запускает симуляцию, собирает статистику.
      - collect_statistics(): объединяет статистику в DataFrame.
      - visualize(): визуализирует динамику очередей.
      - save_results(output_path): сохраняет результаты в CSV.
      - compare_with_traditional(traditional_results): сравнивает с результатами традиционной системы.
    """
    def __init__(self, intersections, simulation_time, dt, traffic_generator):
        self.intersections = intersections
        self.simulation_time = simulation_time
        self.dt = dt
        self.traffic_generator = traffic_generator

        self.time_steps = []
        self.statistics = []

    def run(self):
        current_time = 0.0
        while current_time < self.simulation_time:
            for inter in self.intersections:
                inter.update(self.dt, self.traffic_generator)
            stats = {"time": current_time}
            for inter in self.intersections:
                inter_stats = inter.collect_statistics()
                for direction, data in inter_stats.items():
                    key_queue = f"{inter.intersection_id}_{direction}_queue"
                    key_wait = f"{inter.intersection_id}_{direction}_waiting"
                    stats[key_queue] = data["queue_length"]
                    stats[key_wait] = data["cumulative_waiting_time"]
            self.statistics.append(stats)
            self.time_steps.append(current_time)
            current_time += self.dt
        return self.statistics

    def collect_statistics(self):
        return pd.DataFrame(self.statistics)

    def visualize(self):
        df = self.collect_statistics()
        plt.figure(figsize=(10, 6))
        queue_columns = [col for col in df.columns if col.endswith("_queue")]
        for col in queue_columns:
            plt.plot(df["time"], df[col], label=col)
        plt.xlabel("Время (с)")
        plt.ylabel("Длина очереди")
        plt.title("Динамика длины очередей (умное управление светофорами)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_results(self, output_path):
        df = self.collect_statistics()
        df.to_csv(output_path, index=False)
        print(f"Результаты симуляции сохранены в {output_path}")

    def compare_with_traditional(self, traditional_results):
        smart_df = self.collect_statistics()
        smart_avg = smart_df[[col for col in smart_df.columns if col.endswith("_queue")]].mean()
        trad_avg = traditional_results[[col for col in traditional_results.columns if col.endswith("_queue")]].mean()
        print("Средние длины очередей (умная система):")
        print(smart_avg)
        print("Средние длины очередей (традиционная система):")
        print(trad_avg)

###############################################################################
# Функция main()
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Симуляция умного управления светофорами с учетом плотности потока")
    parser.add_argument("--sim_time", type=float, default=3600,
                        help="Время симуляции в секундах (по умолчанию 3600 сек)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Шаг времени симуляции (по умолчанию 0.1 сек)")
    parser.add_argument("--min_phase_duration", type=float, default=10,
                        help="Минимальная продолжительность фазы (по умолчанию 10 сек)")
    parser.add_argument("--max_phase_duration", type=float, default=60,
                        help="Максимальная продолжительность фазы (по умолчанию 60 сек)")
    parser.add_argument("--yellow_time", type=float, default=3,
                        help="Продолжительность желтого сигнала (по умолчанию 3 сек)")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения результатов симуляции (например, results.csv)")
    args = parser.parse_args()

    # Создаем полосы движения для направлений N, S, E, W
    lanes = {d: Lane(d, length=500) for d in ["N", "S", "E", "W"]}

    # Инициализируем детектор плотности
    detector = TrafficDensityDetector(detector_id="D1", lanes=lanes)

    # Создаем адаптивные фазы: фаза 1 — зеленый для N и S, фаза 2 — зеленый для E и W
    phase1 = AdaptivePhase(phase_id="Phase1", green_directions=["N", "S"],
                           min_duration=args.min_phase_duration, max_duration=args.max_phase_duration,
                           yellow_time=args.yellow_time)
    phase2 = AdaptivePhase(phase_id="Phase2", green_directions=["E", "W"],
                           min_duration=args.min_phase_duration, max_duration=args.max_phase_duration,
                           yellow_time=args.yellow_time)
    phases = [phase1, phase2]

    # Создаем умный контроллер светофоров
    controller = SmartTrafficLightController(controller_id="STLC1", phases=phases,
                                               detector=detector,
                                               min_cycle_time=args.min_phase_duration*2,
                                               max_cycle_time=args.max_phase_duration*2)

    # Формируем перекресток с умным контроллером
    intersection = IntersectionWithSmartLights(intersection_id="I1", controller=controller,
                                                 lanes=lanes, detector=detector)

    # Задаем интенсивности прибытия транспортных средств (транспорт/сек)
    arrival_rates = {"N": 0.2, "S": 0.15, "E": 0.2, "W": 0.15}
    traffic_generator = TrafficGenerator(arrival_rates)

    # Создаем симуляцию
    simulation = SmartTrafficLightSimulation(intersections=[intersection],
                                             simulation_time=args.sim_time,
                                             dt=args.dt,
                                             traffic_generator=traffic_generator)

    print("Запуск симуляции умного управления светофорами...")
    simulation.run()
    print("Симуляция завершена.")

    simulation.visualize()

    if args.output:
        simulation.save_results(args.output)

if __name__ == "__main__":
    main()
