#!/usr/bin/env python3
"""
traffic_simulation.py

Скрипт для моделирования транспортного потока на участке дороги с перекрестками.
Используются библиотеки:
  - numpy для численных вычислений
  - matplotlib для визуализации результатов симуляции
  - pandas для анализа результатов (при необходимости)

Зависимости можно установить командой:
  pip install numpy matplotlib pandas

Пример запуска из командной строки:
  python traffic_simulation.py --sim_time 3600 --dt 0.1 --arrival_rates '{"N":0.2, "S":0.15, "E":0.2, "W":0.15}' --output simulation_results.png

Автор: Islam
Дата: 2025-05-09
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd

###############################################################################
# Класс Vehicle
###############################################################################
class Vehicle:
    """
    Класс для моделирования транспортного средства.

    Атрибуты:
      - vehicle_id: уникальный идентификатор транспортного средства.
      - vehicle_type: тип транспортного средства (например, car, truck, bus, motorcycle).
      - direction: направление движения (N, S, E, W).
      - speed: текущая скорость (м/с).
      - position: положение на полосе (в метрах, начинается с 0).
      - acceleration: ускорение транспортного средства (м/с²).
      - max_speed: максимальная скорость.
      - length: длина транспортного средства (в метрах).
      - waiting_time: накопленное время ожидания (например, при красном свете).
    """

    def __init__(self, vehicle_id, vehicle_type, direction, speed, position, acceleration, max_speed, length):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.direction = direction
        self.speed = speed
        self.position = position
        self.acceleration = acceleration
        self.max_speed = max_speed
        self.length = length
        self.waiting_time = 0.0  # накопленное время ожидания

    def update(self, dt, traffic_light_state, leader_vehicle=None, lane_length=None):
        """
        Обновляет положение и скорость транспортного средства с учетом состояния светофора
        и модели следования за лидером.

        Аргументы:
          dt (float): шаг времени (сек).
          traffic_light_state (str): состояние светофора ("green" или "red") для данного направления.
          leader_vehicle (Vehicle, опционально): транспортное средство, движущееся впереди в очереди.
          lane_length (float, опционально): длина полосы (для учета расстояния до перекрестка).

        Возвращает:
          Словарь с обновленным состоянием транспортного средства.
        """
        # Задаем желаемую скорость равной максимальной, если других ограничений нет
        desired_speed = self.max_speed

        # Если светофор красный, то при приближении к перекрестку транспортное средство должно остановиться
        if traffic_light_state == "red" and lane_length is not None:
            distance_to_intersection = lane_length - (self.position + self.length)
            if distance_to_intersection < 20:  # если расстояние до перекрестка меньше 20 м
                desired_speed = 0

        # Если имеется транспортное средство впереди (лидер), корректируем желаемую скорость для безопасного интервала
        if leader_vehicle is not None:
            gap = leader_vehicle.position - (leader_vehicle.length + self.position)
            safe_gap = 5  # безопасный интервал в метрах
            if gap < safe_gap:
                desired_speed = min(desired_speed, leader_vehicle.speed)

        # Обновляем текущую скорость с использованием ускорения/торможения
        if self.speed < desired_speed:
            self.speed = min(self.speed + self.acceleration * dt, desired_speed, self.max_speed)
        else:
            self.speed = max(self.speed - self.acceleration * dt, desired_speed, 0)

        # Если транспортное средство практически остановилось, увеличиваем время ожидания
        if self.speed < 0.1:
            self.waiting_time += dt

        # Обновляем положение transportного средства
        self.position += self.speed * dt

        return {
            'vehicle_id': self.vehicle_id,
            'vehicle_type': self.vehicle_type,
            'direction': self.direction,
            'position': self.position,
            'speed': self.speed,
            'waiting_time': self.waiting_time
        }

###############################################################################
# Класс TrafficLane
###############################################################################
class TrafficLane:
    """
    Класс для моделирования полосы движения.

    Атрибуты:
      - lane_id: идентификатор полосы.
      - direction: направление полосы (N, S, E, W).
      - length: длина полосы в метрах.
      - vehicles: список транспортных средств, находящихся на полосе.
    """

    def __init__(self, lane_id, direction, length):
        self.lane_id = lane_id
        self.direction = direction
        self.length = length
        self.vehicles = []

    def add_vehicle(self, vehicle):
        """
        Добавляет новое транспортное средство на полосу.
        """
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle):
        """
        Удаляет транспортное средство с полосы.
        """
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)

    def update_vehicles(self, dt, traffic_light_state):
        """
        Обновляет состояние всех транспортных средств на полосе.
        Реализует модель следования за лидером. Если транспортное средство продвинулось дальше конца полосы,
        оно удаляется из списка (т.е. покидает участок моделирования).

        Аргументы:
          dt (float): шаг времени (сек).
          traffic_light_state (str): состояние светофора ("green" или "red") для данной полосы.

        Возвращает:
          Список состояний транспортных средств после обновления.
        """
        # Сортировка транспортных средств по положению: транспортное средство, ближе к перекрестку, имеет большее значение position
        self.vehicles.sort(key=lambda v: v.position)
        updated_states = []

        # Обновляем каждое транспортное средство с учетом поведения лидера (если есть)
        for i, vehicle in enumerate(self.vehicles):
            leader = None
            if i < len(self.vehicles) - 1:
                leader = self.vehicles[i + 1]
            state = vehicle.update(dt, traffic_light_state, leader_vehicle=leader, lane_length=self.length)
            updated_states.append(state)

        # Удаляем транспортные средства, которые покинули полосу (position > length)
        self.vehicles = [v for v in self.vehicles if v.position <= self.length]
        return updated_states

    def get_density(self):
        """
        Возвращает плотность транспортного потока на полосе (количество транспортных средств на метр).
        """
        return len(self.vehicles) / self.length

    def get_queue_length(self):
        """
        Возвращает длину очереди перед светофором.
        Если транспортное средство, находящееся ближе всего к перекрестку, стоит (speed < 0.1 м/с),
        очередь определяется как расстояние от его передней части до конца полосы.
        Если такого транспортного средства нет, возвращается 0.
        """
        threshold = 0.1
        if not self.vehicles:
            return 0
        # Сортируем по позиции по возрастанию; транспортное средство с наибольшим значением position находится ближе к перекрестку
        sorted_vehicles = sorted(self.vehicles, key=lambda v: v.position)
        front_vehicle = sorted_vehicles[-1]
        if front_vehicle.speed < threshold:
            queue_length = self.length - (front_vehicle.position + front_vehicle.length)
            return max(queue_length, 0)
        return 0

###############################################################################
# Класс Intersection
###############################################################################
class Intersection:
    """
    Класс для моделирования перекрестка.

    Атрибуты:
      - intersection_id: идентификатор перекрестка.
      - lanes: словарь полос для каждого направления (например, {"N": lane, "S": lane, ...}).
      - traffic_light_state: словарь состояний светофоров для каждого направления.
    """

    def __init__(self, intersection_id, lanes):
        self.intersection_id = intersection_id
        self.lanes = lanes  # словарь полос по направлениям
        self.traffic_light_state = {}  # словарь, например {"N": "red", "S": "green", ...}

    def add_traffic_light(self, direction, initial_state="red"):
        """
        Добавляет светофор для заданного направления перекрестка.
        """
        self.traffic_light_state[direction] = initial_state

    def set_traffic_light_states(self, states):
        """
        Устанавливает состояния светофоров для перекрестка.
        Аргумент states: словарь, например {"N": "green", "S": "red", ...}
        """
        self.traffic_light_state.update(states)

    def update(self, dt):
        """
        Обновляет состояние транспортных средств на всех полосах перекрестка с использованием
        текущего состояния светофоров для каждого направления.

        Аргумент dt (float): шаг времени симуляции (сек).

        Возвращает:
          Словарь с обновленными состояниями транспортных средств для каждого направления.
        """
        lane_updates = {}
        for direction, lane in self.lanes.items():
            state = self.traffic_light_state.get(direction, "red")
            updated_states = lane.update_vehicles(dt, state)
            lane_updates[direction] = updated_states
        return lane_updates

    def get_statistics(self):
        """
        Собирает статистику по перекрестку:
          - плотность транспортного потока по каждой полосе,
          - длину очереди перед светофором для каждого направления.

        Возвращает:
          Словарь со статистикой.
        """
        stats = {}
        densities = {}
        queue_lengths = {}
        for direction, lane in self.lanes.items():
            densities[direction] = lane.get_density()
            queue_lengths[direction] = lane.get_queue_length()
        stats["densities"] = densities
        stats["queue_lengths"] = queue_lengths
        return stats

###############################################################################
# Класс TrafficGenerator
###############################################################################
class TrafficGenerator:
    """
    Генерирует транспортные средства на основе пуассоновского распределения.

    Атрибуты:
      - arrival_rates: словарь с интенсивностью прибытия транспортных средств для каждого направления,
        например {"N": 0.2, "S": 0.15, "E": 0.2, "W": 0.15}.
      - vehicle_types: распределение вероятностей для типов транспортных средств,
        например {"car": 0.7, "truck": 0.15, "bus": 0.1, "motorcycle": 0.05}.
      - vehicle_params: словарь с предустановленными параметрами для каждого типа транспортного средства.
      - vehicle_id_counter: счетчик для присвоения уникальных идентификаторов.
    """

    def __init__(self, arrival_rates, vehicle_types):
        self.arrival_rates = arrival_rates
        self.vehicle_types = vehicle_types
        self.vehicle_id_counter = 0
        self.vehicle_params = {
            "car": {"acceleration": 2.0, "max_speed": 15.0, "length": 4.5},
            "truck": {"acceleration": 1.0, "max_speed": 10.0, "length": 8.0},
            "bus": {"acceleration": 1.5, "max_speed": 12.0, "length": 12.0},
            "motorcycle": {"acceleration": 2.5, "max_speed": 20.0, "length": 2.5}
        }

    def generate_vehicles(self, dt, current_time):
        """
        Генерирует новые транспортные средства для каждого направления на основании пуассоновского процесса.

        Аргументы:
          dt (float): шаг времени симуляции (сек).
          current_time (float): текущее время симуляции (можно использовать для логирования).

        Возвращает:
          Словарь, где ключ – направление, значение – список новых транспортных средств.
        """
        new_vehicles = {direction: [] for direction in self.arrival_rates.keys()}
        for direction, rate in self.arrival_rates.items():
            # Число транспортных средств, прибывающих за интервал dt, согласно пуассоновскому распределению
            num_arrivals = np.random.poisson(rate * dt)
            for _ in range(num_arrivals):
                # Выбираем тип транспортного средства согласно заданному распределению
                types = list(self.vehicle_types.keys())
                probabilities = list(self.vehicle_types.values())
                chosen_type = np.random.choice(types, p=probabilities)
                params = self.vehicle_params.get(chosen_type, {"acceleration": 2.0, "max_speed": 15.0, "length": 4.5})
                # Создаем транспортное средство с начальной позицией 0 на полосе
                vehicle = Vehicle(
                    vehicle_id=self.vehicle_id_counter,
                    vehicle_type=chosen_type,
                    direction=direction,
                    speed=0.0,           # начинаем с нулевой скорости
                    position=0.0,        # начальная позиция на полосе
                    acceleration=params["acceleration"],
                    max_speed=params["max_speed"],
                    length=params["length"]
                )
                self.vehicle_id_counter += 1
                new_vehicles[direction].append(vehicle)
        return new_vehicles

###############################################################################
# Класс TrafficSimulation
###############################################################################
class TrafficSimulation:
    """
    Основной класс симуляции транспортного потока.

    Атрибуты:
      - intersections: список объектов Intersection для моделирования.
      - traffic_generator: экземпляр класса TrafficGenerator.
      - simulation_time: общее время симуляции (сек).
      - dt: шаг времени симуляции (сек).
      - statistics: список для сохранения статистических данных по времени.
      - time_steps: список временных меток для построения графиков.
    """

    def __init__(self, intersections, traffic_generator, simulation_time, dt):
        self.intersections = intersections
        self.traffic_generator = traffic_generator
        self.simulation_time = simulation_time
        self.dt = dt
        self.statistics = []
        self.time_steps = []

    def run(self):
        """
        Запускает симуляцию:
          - На каждом шаге генерируются новые транспортные средства.
          - Обновляются состояния светофоров по циклическому режиму (например, каждые 30 секунд меняются фазы).
          - Обновляются состояния транспортных средств на перекрестке.
          - Собирается статистика по плотности потока и очередям.
        
        Возвращает собранную статистику.
        """
        current_time = 0.0
        # Задаем период переключения фаз светофора (например, каждые 30 сек)
        light_cycle = 30  
        while current_time < self.simulation_time:
            # Для простоты: в одной фазе зелёным являются N и S, в другой – E и W
            mode = int((current_time // light_cycle) % 2)
            for intersection in self.intersections:
                if mode == 0:
                    intersection.set_traffic_light_states({"N": "green", "S": "green", "E": "red", "W": "red"})
                else:
                    intersection.set_traffic_light_states({"N": "red", "S": "red", "E": "green", "W": "green"})

            # Генерация новых транспортных средств
            new_vehicles = self.traffic_generator.generate_vehicles(self.dt, current_time)

            # Добавляем новые транспортные средства на полосы первого перекрестка
            intersection = self.intersections[0]
            for direction, vehicles in new_vehicles.items():
                lane = intersection.lanes.get(direction)
                if lane is not None:
                    for vehicle in vehicles:
                        lane.add_vehicle(vehicle)

            # Обновляем состояния транспортных средств на перекрестке
            intersection.update(self.dt)

            # Собираем статистику по перекрестку
            stats = intersection.get_statistics()
            stats["time"] = current_time
            self.statistics.append(stats)
            self.time_steps.append(current_time)

            current_time += self.dt

        return self.statistics

    def visualize(self, output_path=None):
        """
        Визуализирует результаты симуляции:
          - График средней плотности транспортного потока.
          - График длины очередей для каждого направления.

        Аргументы:
          output_path (str, опционально): путь для сохранения графика.
        """
        times = [stat["time"] for stat in self.statistics]
        avg_densities = []
        queue_N = []
        queue_S = []
        queue_E = []
        queue_W = []

        for stat in self.statistics:
            densities = stat["densities"]
            avg_density = np.mean(list(densities.values()))
            avg_densities.append(avg_density)
            queues = stat["queue_lengths"]
            queue_N.append(queues.get("N", 0))
            queue_S.append(queues.get("S", 0))
            queue_E.append(queues.get("E", 0))
            queue_W.append(queues.get("W", 0))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(times, avg_densities, label="Average Density", color="blue")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Density (veh/m)")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_title("Average Traffic Density Over Time")

        axs[1].plot(times, queue_N, label="North", color="green")
        axs[1].plot(times, queue_S, label="South", color="red")
        axs[1].plot(times, queue_E, label="East", color="purple")
        axs[1].plot(times, queue_W, label="West", color="orange")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Queue Length (m)")
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_title("Queue Lengths Over Time")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Результаты симуляции сохранены в {output_path}")
        plt.show()

###############################################################################
# Функция main
###############################################################################
def main():
    """
    Главная функция, которая:
      1. Парсит параметры командной строки.
      2. Создает экземпляры необходимых классов (TrafficGenerator, Intersection, TrafficSimulation).
      3. Запускает симуляцию.
      4. Визуализирует и (опционально) сохраняет результаты.
    """
    parser = argparse.ArgumentParser(description="Моделирование транспортного потока с перекрестками")
    parser.add_argument("--sim_time", type=float, default=3600,
                        help="Общее время симуляции в секундах, по умолчанию 3600 (1 час)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Шаг времени симуляции в секундах, по умолчанию 0.1")
    parser.add_argument("--arrival_rates", type=str, default='{"N":0.2, "S":0.15, "E":0.2, "W":0.15}',
                        help="Интенсивность прибытия транспортных средств (JSON строка). Например: '{\"N\":0.2, \"S\":0.15, \"E\":0.2, \"W\":0.15}'")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения результатов симуляции (например, simulation_results.png)")
    args = parser.parse_args()

    arrival_rates = json.loads(args.arrival_rates)
    # Определяем распределение типов транспортных средств (при необходимости можно изменить)
    vehicle_types = {"car": 0.7, "truck": 0.15, "bus": 0.1, "motorcycle": 0.05}

    # Создаем один перекресток со 4 полосами.
    # Для каждой полосы задаем длину (например, 500 метров)
    lane_length = 500.0
    lanes = {
        "N": TrafficLane(lane_id="N1", direction="N", length=lane_length),
        "S": TrafficLane(lane_id="S1", direction="S", length=lane_length),
        "E": TrafficLane(lane_id="E1", direction="E", length=lane_length),
        "W": TrafficLane(lane_id="W1", direction="W", length=lane_length)
    }

    # Создаем объект Intersection и добавляем светофоры для каждого направления
    intersection = Intersection(intersection_id="I1", lanes=lanes)
    for direction in lanes.keys():
        intersection.add_traffic_light(direction, initial_state="red")

    # Создаем генератор транспортных средств
    traffic_generator = TrafficGenerator(arrival_rates, vehicle_types)

    # Создаем симуляцию с одним перекрестком
    simulation = TrafficSimulation(
        intersections=[intersection],
        traffic_generator=traffic_generator,
        simulation_time=args.sim_time,
        dt=args.dt
    )

    print("Начало симуляции...")
    simulation.run()
    print("Симуляция завершена.")

    # Визуализируем и (при необходимости) сохраняем результаты
    simulation.visualize(output_path=args.output)

if __name__ == "__main__":
    main()
