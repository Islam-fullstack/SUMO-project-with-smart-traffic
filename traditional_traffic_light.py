#!/usr/bin/env python3
"""
traditional_traffic_light.py

Скрипт для моделирования традиционного алгоритма работы светофоров с фиксированными фазами.
Используемые библиотеки:
  - numpy для обработки числовых данных
  - matplotlib для визуализации результатов симуляции
  - pandas для анализа и сохранения статистики

Установка зависимостей:
  pip install numpy matplotlib pandas

Пример запуска из командной строки:
  python traditional_traffic_light.py --sim_time 3600 --dt 0.1 --phase1_duration 30 --phase2_duration 30 --yellow_time 3 --output results.csv

Автор: Islam
Дата: 2025-05-09
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

###############################################################################
# Класс TrafficLightPhase
###############################################################################
class TrafficLightPhase:
    """
    Класс TrafficLightPhase моделирует отдельную фазу работы светофора.

    Аргументы:
      - phase_id: уникальный идентификатор фазы.
      - green_directions: список направлений (например, ["N", "S"]), которым в данной фазе
                          предоставляется зелёный сигнал.
      - duration: продолжительность фазы (секунды).
      - yellow_time: время желтого сигнала в конце фазы (секунды, по умолчанию 3).
    """
    def __init__(self, phase_id, green_directions, duration, yellow_time=3):
        self.phase_id = phase_id
        self.green_directions = green_directions
        self.duration = duration
        self.yellow_time = yellow_time

    def get_state(self, elapsed_time):
        """
        Возвращает текущее состояние светофора для всех направлений с учетом желтого сигнала.
        Если оставшееся время фазы меньше yellow_time, для зеленых направлений возвращается "yellow".
        
        Аргументы:
          - elapsed_time: время, прошедшее с начала данной фазы.
          
        Возвращает:
          Словарь состояний для направлений, например:
          {"N": "green", "S": "green", "E": "red", "W": "red"}
        """
        # Инициализируем состояния для всех направлений (N, S, E, W)
        state = {}
        for d in ["N", "S", "E", "W"]:
            if d in self.green_directions:
                # Если осталось меньше времени для переключения, включаем желтый
                if elapsed_time >= self.duration - self.yellow_time:
                    state[d] = "yellow"
                else:
                    state[d] = "green"
            else:
                state[d] = "red"
        return state

###############################################################################
# Класс TraditionalTrafficLightController
###############################################################################
class TraditionalTrafficLightController:
    """
    Контроллер управляет переключением фаз светофоров с фиксированной длительностью.

    Аргументы:
      - controller_id: уникальный идентификатор контроллера.
      - phases: список объектов TrafficLightPhase.
      - cycle_time: общее время цикла (сумма продолжительностей всех фаз).
      
    Методы:
      - update(dt): обновляет внутренний таймер контроллера.
      - get_current_phase(): возвращает текущую активную фазу и локальное время в ней.
      - get_current_state(): возвращает текущее состояние светофора для всех направлений.
      - get_remaining_time(): возвращает оставшееся время текущей фазы.
      - reset(): сбрасывает контроллер в начальное состояние.
    """
    def __init__(self, controller_id, phases, cycle_time):
        self.controller_id = controller_id
        self.phases = phases
        self.cycle_time = cycle_time
        self.current_time = 0.0  # время, прошедшее с начала цикла

    def update(self, dt):
        """
        Обновляет внутренний таймер контроллера на значение dt.
        Если время цикла превышено, происходит переход на новую итерацию цикла.
        """
        self.current_time += dt
        if self.current_time >= self.cycle_time:
            self.current_time = self.current_time % self.cycle_time

    def get_current_phase(self):
        """
        Определяет текущую фазу, исходя из накопленного времени.
        
        Возвращает:
          Кортеж (текущая фаза, локальное время в фазе).
        """
        elapsed = 0.0
        for phase in self.phases:
            if self.current_time < elapsed + phase.duration:
                local_time = self.current_time - elapsed
                return phase, local_time
            elapsed += phase.duration
        # Если не найдено (на всякий случай)
        return self.phases[-1], self.current_time - (elapsed - self.phases[-1].duration)

    def get_current_state(self):
        """
        Возвращает текущее состояние светофора для всех направлений.
        """
        phase, local_time = self.get_current_phase()
        return phase.get_state(local_time)

    def get_remaining_time(self):
        """
        Возвращает оставшееся время текущей фазы.
        """
        phase, local_time = self.get_current_phase()
        return phase.duration - local_time

    def reset(self):
        """
        Сбрасывает контроллер в начальное состояние.
        """
        self.current_time = 0.0

###############################################################################
# Класс Lane
###############################################################################
class Lane:
    """
    Класс моделирует полосу движения по направлению.

    Атрибуты:
      - direction: направление движения ("N", "S", "E" или "W").
      - queue: количество транспортных средств, ожидающих в очереди.
      - cumulative_waiting_time: накопленное время ожидания транспортных средств.
    
    Метод update() обновляет состояние полосы с учетом новых прибытия и расхода при
    зелёном сигнале.
    """
    def __init__(self, direction):
        self.direction = direction
        self.queue = 0
        self.cumulative_waiting_time = 0.0

    def update(self, dt, arrivals, traffic_light_state, departure_rate=1.0):
        """
        Обновляет полосу:
          - Прибывшие транспортные средства (arrivals) добавляются в очередь.
          - Накопленное время ожидания увеличивается пропорционально текущей длине очереди.
          - Если сигнал "green", транспортные средства покидают очередь с заданной интенсивностью.
        
        Аргументы:
          - dt: шаг времени симуляции.
          - arrivals: число транспортных средств, прибывших за интервал dt.
          - traffic_light_state: состояние светофора для данного направления ("green", "yellow" или "red").
          - departure_rate: среднее число транспортных средств, покидающих очередь в секунду при зелёном сигнале.
        
        Возвращает:
          Текущее количество транспортных средств в очереди.
        """
        # Добавляем новые транспортные средства
        self.queue += arrivals
        # Накопление времени ожидания на всей очереди
        self.cumulative_waiting_time += self.queue * dt
        # Если сигнал зеленый, из очереди уходят транспортные средства
        if traffic_light_state == "green":
            departed = np.random.poisson(departure_rate * dt)
            departed = min(self.queue, departed)
            self.queue -= departed
        # При желтом сигнале транспортные средства не стартуют – считаем как остановку
        return self.queue

###############################################################################
# Класс IntersectionWithTraditionalLights
###############################################################################
class IntersectionWithTraditionalLights:
    """
    Представляет перекресток с традиционным контроллером светофоров.

    Аргументы:
      - intersection_id: идентификатор перекрестка.
      - controller: экземпляр TraditionalTrafficLightController.
      - lanes: словарь с объектами Lane для каждого направления.

    Методы:
      - update(dt, traffic_generator): обновляет состояние светофоров и очереди на полосах.
      - get_traffic_light_states(): возвращает текущее состояние светофоров.
      - collect_statistics(): собирает статистику по очередям и времени ожидания.
    """
    def __init__(self, intersection_id, controller, lanes):
        self.intersection_id = intersection_id
        self.controller = controller
        self.lanes = lanes  # например, {"N": lane_obj, "S": lane_obj, ...}

    def update(self, dt, traffic_generator):
        """
        Обновляет состояние контроллера и полос перекрестка.
        Новые транспортные средства генерируются с помощью traffic_generator.
        
        Аргументы:
          - dt: шаг времени симуляции.
          - traffic_generator: объект для генерации прибытия транспортных средств.
        """
        # Обновляем контроллер светофора
        self.controller.update(dt)
        # Получаем текущее состояние светофоров
        tl_states = self.controller.get_current_state()
        # Генерируем прибытия для каждого направления
        new_arrivals = traffic_generator.generate(dt)
        # Обновляем каждую полосу с учетом прибытия и состояния светофора
        for direction, lane in self.lanes.items():
            arrivals = new_arrivals.get(direction, 0)
            lane.update(dt, arrivals, tl_states.get(direction, "red"))

    def get_traffic_light_states(self):
        """
        Возвращает текущее состояние светофоров на перекрестке.
        """
        return self.controller.get_current_state()

    def collect_statistics(self):
        """
        Собирает статистику по каждой полосе: длину очереди и накопленное время ожидания.
        
        Возвращает:
          Словарь со статистикой вида:
          { "N": {"queue_length": ..., "cumulative_waiting_time": ...}, ... }
        """
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
    Генерирует прибытия транспортных средств для каждого направления на основе пуассоновского процесса.

    Аргументы:
      - arrival_rates: словарь интенсивностей прибытия (транспорт/сек), например: {"N": 0.2, "S": 0.2, "E": 0.2, "W": 0.2}
    """
    def __init__(self, arrival_rates):
        self.arrival_rates = arrival_rates

    def generate(self, dt):
        """
        Для каждого направления генерирует число прибывших транспортных средств за интервал dt.
        
        Возвращает:
          Словарь вида { "N": число, "S": число, ... }
        """
        return {direction: np.random.poisson(rate * dt)
                for direction, rate in self.arrival_rates.items()}

###############################################################################
# Класс TraditionalTrafficLightSimulation
###############################################################################
class TraditionalTrafficLightSimulation:
    """
    Класс для симуляции работы традиционных светофоров на перекрестке с фиксированными фазами.

    Аргументы:
      - intersections: список объектов IntersectionWithTraditionalLights.
      - simulation_time: общее время симуляции (сек).
      - dt: шаг времени симуляции (сек).
      - traffic_generator: экземпляр TrafficGenerator для генерации прибытия.
    
    Методы:
      - run(): запускает симуляцию и собирает статистику.
      - collect_statistics(): возвращает статистику симуляции в виде DataFrame.
      - visualize(): визуализирует результаты (например, динамику длины очередей).
      - save_results(output_path): сохраняет статистику в CSV-файл.
    """
    def __init__(self, intersections, simulation_time, dt, traffic_generator):
        self.intersections = intersections
        self.simulation_time = simulation_time
        self.dt = dt
        self.traffic_generator = traffic_generator
        self.time_steps = []
        self.statistics = []

    def run(self):
        """
        Запускает симуляцию: на каждом шаге обновляются состояния перекрестков, собирается статистика.
        """
        current_time = 0.0
        while current_time < self.simulation_time:
            # Обновляем каждое пересечение
            for inter in self.intersections:
                inter.update(self.dt, self.traffic_generator)
            # Собираем стат. данные: для каждого перекрестка сохраняем длину очередей и общее время ожидания
            stats = {"time": current_time}
            for inter in self.intersections:
                inter_stats = inter.collect_statistics()
                for direction, s in inter_stats.items():
                    stats[f"{inter.intersection_id}_{direction}_queue"] = s["queue_length"]
                    stats[f"{inter.intersection_id}_{direction}_waiting"] = s["cumulative_waiting_time"]
            self.statistics.append(stats)
            self.time_steps.append(current_time)
            current_time += self.dt
        return self.statistics

    def collect_statistics(self):
        """
        Объединяет собранную статистику в DataFrame.
        """
        return pd.DataFrame(self.statistics)

    def visualize(self):
        """
        Визуализирует динамику длины очередей по всем направлениям.
        """
        df = self.collect_statistics()
        plt.figure(figsize=(10, 6))
        # Отбираем столбцы, содержащие информацию о длине очередей
        queue_columns = [col for col in df.columns if col.endswith("_queue")]
        for col in queue_columns:
            plt.plot(df["time"], df[col], label=col)
        plt.xlabel("Время (с)")
        plt.ylabel("Длина очереди (транспорт. единиц)")
        plt.title("Динамика длины очередей на перекрестке")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_results(self, output_path):
        """
        Сохраняет собранную статистику в CSV-файл.
        
        Аргументы:
          - output_path: путь для сохранения результатов.
        """
        df = self.collect_statistics()
        df.to_csv(output_path, index=False)
        print(f"Результаты симуляции сохранены в {output_path}")

###############################################################################
# Функция main()
###############################################################################
def main():
    """
    Главная функция:
      1. Парсит параметры командной строки.
      2. Создаёт фазы светофора (фаза 1: зеленый для N и S, фаза 2: зеленый для E и W).
      3. Формирует контроллер, полосы и перекресток.
      4. Создаёт генератор транспортных средств.
      5. Запускает симуляцию, визуализирует результаты и (при необходимости) сохраняет их.
    """
    parser = argparse.ArgumentParser(description="Симуляция работы традиционных светофоров с фиксированными фазами")
    parser.add_argument("--sim_time", type=float, default=3600,
                        help="Общее время симуляции в секундах, по умолчанию 3600 (1 час)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Шаг времени симуляции в секундах, по умолчанию 0.1")
    parser.add_argument("--phase1_duration", type=float, default=30,
                        help="Продолжительность первой фазы (зелёный для N и S), по умолчанию 30 секунд")
    parser.add_argument("--phase2_duration", type=float, default=30,
                        help="Продолжительность второй фазы (зелёный для E и W), по умолчанию 30 секунд")
    parser.add_argument("--yellow_time", type=float, default=3,
                        help="Продолжительность желтого сигнала, по умолчанию 3 секунды")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения результатов симуляции (например, results.csv)")
    args = parser.parse_args()

    # Создаем фазы светофора:
    phase1 = TrafficLightPhase(phase_id="Phase1", green_directions=["N", "S"],
                               duration=args.phase1_duration, yellow_time=args.yellow_time)
    phase2 = TrafficLightPhase(phase_id="Phase2", green_directions=["E", "W"],
                               duration=args.phase2_duration, yellow_time=args.yellow_time)
    phases = [phase1, phase2]
    cycle_time = args.phase1_duration + args.phase2_duration

    # Создаем контроллер светофора:
    controller = TraditionalTrafficLightController(controller_id="TLC1", phases=phases, cycle_time=cycle_time)

    # Создаем полосы движения для каждого направления:
    lanes = {d: Lane(d) for d in ["N", "S", "E", "W"]}

    # Инициализируем перекресток с традиционными светофорами:
    intersection = IntersectionWithTraditionalLights(intersection_id="I1", controller=controller, lanes=lanes)

    # Задаем интенсивности прибытия транспортных средств (число транспорт. единиц/сек)
    arrival_rates = {"N": 0.2, "S": 0.2, "E": 0.2, "W": 0.2}
    traffic_generator = TrafficGenerator(arrival_rates)

    # Создаем симуляцию:
    simulation = TraditionalTrafficLightSimulation(
        intersections=[intersection],
        simulation_time=args.sim_time,
        dt=args.dt,
        traffic_generator=traffic_generator
    )

    print("Запуск симуляции...")
    simulation.run()
    print("Симуляция завершена.")

    # Визуализируем результаты
    simulation.visualize()

    # Сохраняем результаты, если указан output
    if args.output:
        simulation.save_results(args.output)

if __name__ == "__main__":
    main()
