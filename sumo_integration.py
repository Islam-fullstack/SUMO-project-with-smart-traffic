#!/usr/bin/env python3
"""
sumo_integration.py

Интеграция разработанной системы умных светофоров с микроскопическим симулятором 
дорожного движения SUMO (Simulation of Urban MObility). Система использует данные OSM 
и работает с перекрестками, которые уже созданы в проекте SUMO.

Зависимости:
  - sumolib
  - traci
  - numpy
  - pandas
  - matplotlib

Установка зависимостей:
  pip install sumolib traci numpy pandas matplotlib

Пример запуска из командной строки:
  python sumo_integration.py --sumo_config osm.sumocfg --gui True --steps 3600 --min_phase 10 --max_phase 60 --output_dir ./sumo_results

Автор: Islam
Дата: 2025-05-09
"""

import os
import sys
import subprocess
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Импорт модулей SUMO (traci и sumolib)
import traci
import sumolib

###############################################################################
# Класс SUMOConnection
###############################################################################
class SUMOConnection:
    """
    Устанавливает соединение с симулятором SUMO через TraCI.
    
    Параметры:
      - sumo_config_path: путь к конфигурационному файлу SUMO (.sumocfg)
      - use_gui: использовать ли графический интерфейс SUMO (True/False)
      - port: порт для соединения с TraCI (по умолчанию 8813)
    """
    def __init__(self, sumo_config_path, use_gui=True, port=8813):
        self.sumo_config_path = sumo_config_path
        self.use_gui = use_gui
        self.port = port
        self.sumo_process = None

    def start_simulation(self):
        """
        Запускает симуляцию SUMO с использованием TraCI. Если use_gui=True, используется
        sumo-gui, иначе – sumo.
        """
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_config_path, "--remote-port", str(self.port)]
        print("Запуск SUMO с командой:", " ".join(sumo_cmd))
        self.sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Даем время на запуск SUMO
        time.sleep(2)
        traci.init(port=self.port)
        print("SUMO запущен и TraCI подключен.")

    def stop_simulation(self):
        """
        Останавливает симуляцию и закрывает соединение TraCI.
        """
        traci.close()
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()
        print("SUMO остановлен.")

    def get_step(self):
        """
        Возвращает текущий шаг симуляции.
        """
        return traci.simulation.getTime()

    def step(self):
        """
        Выполняет один шаг симуляции.
        """
        traci.simulationStep()

    def get_traffic_lights(self):
        """
        Возвращает список идентификаторов светофоров в симуляции.
        """
        return traci.trafficlight.getIDList()

    def get_lane_ids(self):
        """
        Возвращает список идентификаторов полос в симуляции.
        """
        return traci.lane.getIDList()

    def get_edges(self):
        """
        Возвращает список идентификаторов ребер дорожной сети.
        """
        return traci.edge.getIDList()

###############################################################################
# Класс SUMOTrafficDetector
###############################################################################
class SUMOTrafficDetector:
    """
    Детектор транспортного потока для конкретного перекрестка (junction).

    Параметры:
      - sumo_connection: экземпляр SUMOConnection
      - junction_id: идентификатор перекрестка
      - detector_range: расстояние детектирования транспортных средств (в метрах)
    """
    def __init__(self, sumo_connection, junction_id, detector_range):
        self.sumo_connection = sumo_connection
        self.junction_id = junction_id
        self.detector_range = detector_range
        self.incoming_lanes = []  # Список идентификаторов входящих полос (будет заполнен в setup_detectors)
        self.detection_data = {}  # Словарь для хранения последних данных

    def setup_detectors(self):
        """
        Создает детекторы на всех входящих ребрах перекрестка.
        Здесь в качестве упрощения детектором рассматриваются все полосы, входящие в перекресток.
        Реальная реализация может использовать SUMO inductionloop detectors, определенные в сети.
        """
        # Найдем все полосы, смежные с перекрестком.
        # Используем API traci для получения информации об окружении.
        all_lanes = traci.lane.getIDList()
        self.incoming_lanes = [lane for lane in all_lanes if self.junction_id in lane]
        print(f"Для перекрестка {self.junction_id} определены детекторы для полос: {self.incoming_lanes}")

    def get_vehicle_counts(self):
        """
        Возвращает количество транспортных средств для каждой входящей полосы.
        """
        counts = {}
        for lane in self.incoming_lanes:
            counts[lane] = traci.lane.getLastStepVehicleNumber(lane)
        return counts

    def get_vehicle_speeds(self):
        """
        Возвращает средние скорости транспортных средств на каждой входящей полосе.
        """
        speeds = {}
        for lane in self.incoming_lanes:
            speeds[lane] = traci.lane.getLastStepMeanSpeed(lane)
        return speeds

    def get_waiting_times(self):
        """
        Возвращает общее время ожидания транспортных средств на каждой входящей полосе.
        """
        waiting = {}
        for lane in self.incoming_lanes:
            waiting[lane] = traci.lane.getWaitingTime(lane)
        return waiting

    def get_queue_lengths(self):
        """
        Возвращает длину очередей на каждой входящей полосе.
        """
        # Для упрощения используем количество транспортных средств как длину очереди.
        # Возможна более сложная оценка (например, суммарная длина транспортных средств).
        return self.get_vehicle_counts()

    def get_traffic_density(self):
        """
        Возвращает плотность транспортного потока для каждой входящей полосы: 
        количество транспортных средств, деленное на длину полосы.
        """
        density = {}
        for lane in self.incoming_lanes:
            veh_num = traci.lane.getLastStepVehicleNumber(lane)
            lane_length = traci.lane.getLength(lane)
            density[lane] = veh_num / lane_length if lane_length > 0 else 0
        return density

    def update(self):
        """
        Обновляет данные детектора. Результаты сохраняются в self.detection_data.
        """
        self.detection_data = {
            "vehicle_counts": self.get_vehicle_counts(),
            "vehicle_speeds": self.get_vehicle_speeds(),
            "waiting_times": self.get_waiting_times(),
            "queue_lengths": self.get_queue_lengths(),
            "traffic_density": self.get_traffic_density()
        }
        # Вывод для отладки
        # print(f"Обновленные данные детектора для {self.junction_id}: {self.detection_data}")

    def export_detection_data(self, output_path):
        """
        Экспортирует текущие данные детектора в файл CSV.
        """
        df = pd.DataFrame(self.detection_data)
        df.to_csv(output_path, index=False)
        print(f"Данные детектора экспортированы в {output_path}")

###############################################################################
# Класс SUMOTrafficLightController
###############################################################################
class SUMOTrafficLightController:
    """
    Контроллер светофоров в SUMO. Может работать как в режиме традиционного, так и умного алгоритма.

    Параметры:
      - sumo_connection: экземпляр SUMOConnection
      - traffic_light_id: идентификатор светофора в SUMO
      - detector: экземпляр SUMOTrafficDetector
      - controller_type: тип контроллера ("traditional" или "smart")
    """
    def __init__(self, sumo_connection, traffic_light_id, detector, controller_type="traditional"):
        self.sumo_connection = sumo_connection
        self.traffic_light_id = traffic_light_id
        self.detector = detector
        self.controller_type = controller_type

    def get_current_phase(self):
        """
        Возвращает текущую фазу светофора.
        """
        return traci.trafficlight.getPhase(self.traffic_light_id)

    def set_phase(self, phase_index):
        """
        Устанавливает фазу светофора.
        """
        traci.trafficlight.setPhase(self.traffic_light_id, phase_index)
        print(f"Светофор {self.traffic_light_id}: установлена фаза {phase_index}")

    def get_phase_duration(self, phase_index):
        """
        Возвращает продолжительность указанной фазы.
        Здесь используется функция traci.trafficlight.getNextSwitch, как пример.
        """
        # Пример: возвращаем время до следующего переключения
        return traci.trafficlight.getNextSwitch(self.traffic_light_id)

    def set_phase_duration(self, phase_index, duration):
        """
        Устанавливает продолжительность указанной фазы. Если API SUMO не позволяет менять длительность
        напрямую, можно формировать новую программу.
        """
        traci.trafficlight.setPhaseDuration(self.traffic_light_id, duration)
        print(f"Светофор {self.traffic_light_id}: длительность фазы {phase_index} установлена в {duration} сек")

    def get_program(self):
        """
        Возвращает текущую программу светофора.
        """
        return traci.trafficlight.getCompleteRedYellowGreenDefinition(self.traffic_light_id)

    def set_program(self, program_id):
        """
        Устанавливает программу светофора.
        """
        traci.trafficlight.setProgram(self.traffic_light_id, program_id)
        print(f"Светофор {self.traffic_light_id}: установлена программа {program_id}")

    def create_traditional_program(self):
        """
        Создает традиционную программу с фиксированными фазами.
        Здесь можно задать список фиксированных фаз и переключать их циклически.
        """
        # Пример – использование предопределенной программы из файловой системы SUMO
        program = "static"  # имя программы; может быть определено в конфигурации сети
        self.set_program(program)
        print(f"Светофор {self.traffic_light_id}: создана традиционная программа")

    def create_smart_program(self):
        """
        Создает умную программу с адаптивными фазами.
        Реализация будет зависеть от логики изменения длительности фаз.
        """
        # Здесь можно задать базовую программу и далее изменять длительности фаз через set_phase_duration
        program = "actuated"  # или любое имя программы, поддерживающее адаптацию
        self.set_program(program)
        print(f"Светофор {self.traffic_light_id}: создана умная программа")

    def update(self, step):
        """
        Обновляет состояние контроллера на основе текущего шага симуляции и данных детектора.
        Реализует логику переключения фаз в зависимости от выбранного алгоритма.
        """
        # Пример: если умный контроллер, проверяем данные детектора и переключаем фазы.
        if self.controller_type == "smart":
            self.detector.update()
            waiting = self.detector.get_waiting_times()
            # Если суммарное время ожидания по входящим полосам превышает порог, переключаем фазу.
            total_wait = sum(waiting.values())
            if total_wait > 20:
                current_phase = self.get_current_phase()
                next_phase = (current_phase + 1) % 4  # предположим 4 фазы
                self.set_phase(next_phase)
        else:
            # Традиционный алгоритм может работать циклически с фиксированным интервалом.
            cycle_time = 30  # например, каждые 30 сек
            if step % cycle_time == 0:
                current_phase = self.get_current_phase()
                next_phase = (current_phase + 1) % 4
                self.set_phase(next_phase)

###############################################################################
# Класс SUMOTrafficLightOptimizer
###############################################################################
class SUMOTrafficLightOptimizer:
    """
    Оптимизатор для умных светофоров. На основе данных детектора оптимизирует
    продолжительность фаз и выбирает следующую фазу.
    
    Параметры:
      - controller: экземпляр SUMOTrafficLightController
      - detector: экземпляр SUMOTrafficDetector
      - min_phase_duration: минимальная длительность фазы (сек)
      - max_phase_duration: максимальная длительность фазы (сек)
      - optimization_interval: интервал оптимизации (в шагах симуляции)
    """
    def __init__(self, controller, detector, min_phase_duration, max_phase_duration, optimization_interval):
        self.controller = controller
        self.detector = detector
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        self.optimization_interval = optimization_interval
        self.last_optimization_step = 0

    def calculate_phase_score(self, phase_index):
        """
        Рассчитывает оценку эффективности фазы. В данном примере суммируются
        количество транспортных средств и время ожидания на соответствующих входящих полосах.
        """
        self.detector.update()
        counts = self.detector.get_vehicle_counts()
        waiting = self.detector.get_waiting_times()
        score = sum(counts.values()) + sum(waiting.values()) * 0.5
        print(f"Оценка фазы {phase_index}: {score}")
        return score

    def optimize_phase_durations(self):
        """
        Оптимизирует продолжительность фаз на основе данных детектора. Если трафик высокий – 
        увеличивает длительность, если низкий – уменьшает.
        """
        score = self.calculate_phase_score(self.controller.get_current_phase())
        # Простой линейный алгоритм: interpolate между min и max длительностью по нормализованному баллу.
        normalized = min(score / 50.0, 1.0)
        new_duration = self.min_phase_duration + (self.max_phase_duration - self.min_phase_duration) * normalized
        current_phase = self.controller.get_current_phase()
        self.controller.set_phase_duration(current_phase, new_duration)
        print(f"Оптимизированная длительность фазы {current_phase}: {new_duration} сек")

    def select_next_phase(self):
        """
        Выбирает следующую фазу на основе данных детектора.
        """
        # Простой пример: выбираем фазу с наибольшей суммарной оценкой.
        scores = []
        for phase in range(4):  # предполагается 4 фазы
            score = self.calculate_phase_score(phase)
            scores.append(score)
        next_phase = np.argmax(scores)
        print(f"Выбрана следующая фаза: {next_phase}")
        return next_phase

    def update(self, step):
        """
        Обновляет оптимизатор. Если прошёл заданный интервал, оптимизирует параметры.
        """
        if step - self.last_optimization_step >= self.optimization_interval:
            self.optimize_phase_durations()
            self.last_optimization_step = step

    def apply_optimization(self):
        """
        Применяет оптимизированные параметры к контроллеру.
        """
        next_phase = self.select_next_phase()
        self.controller.set_phase(next_phase)

###############################################################################
# Класс SUMOSimulationRunner
###############################################################################
class SUMOSimulationRunner:
    """
    Запускает симуляцию SUMO с интеграцией контроллеров, детекторов и оптимизаторов светофоров.
    
    Параметры:
      - sumo_connection: экземпляр SUMOConnection
      - controllers: список экземпляров SUMOTrafficLightController
      - detectors: список экземпляров SUMOTrafficDetector
      - optimizers: список экземпляров SUMOTrafficLightOptimizer
      - simulation_steps: количество шагов симуляции
      - output_dir: директория для сохранения результатов
    """
    def __init__(self, sumo_connection, controllers, detectors, optimizers, simulation_steps, output_dir):
        self.sumo_connection = sumo_connection
        self.controllers = controllers
        self.detectors = detectors
        self.optimizers = optimizers
        self.simulation_steps = simulation_steps
        self.output_dir = output_dir
        self.statistics = []  # Для сбора статистики по шагам

    def run_simulation(self):
        """
        Запускает симуляцию на заданное количество шагов. На каждом шаге обновляются
        контроллеры и оптимизаторы, а также собираются данные с детекторов.
        """
        for step in range(self.simulation_steps):
            # Выполняем один шаг SUMO
            self.sumo_connection.step()
            current_time = self.sumo_connection.get_step()
            # Обновляем контроллеры и оптимизаторы
            for controller in self.controllers:
                controller.update(step)
            for optimizer in self.optimizers:
                optimizer.update(step)
            # Обновляем все детекторы
            for detector in self.detectors:
                detector.update()
            # Собираем статистику (например, суммарное время ожидания на всех детекторах)
            stats = {"step": step, "time": current_time}
            for detector in self.detectors:
                waiting = detector.get_waiting_times()
                stats[f"{detector.junction_id}_total_waiting"] = sum(waiting.values())
            self.statistics.append(stats)
        print("Симуляция завершена.")

    def collect_statistics(self):
        """
        Собирает статистику в DataFrame.
        """
        df = pd.DataFrame(self.statistics)
        return df

    def save_results(self):
        """
        Сохраняет результаты симуляции в CSV-файл.
        """
        df = self.collect_statistics()
        output_path = os.path.join(self.output_dir, "simulation_statistics.csv")
        df.to_csv(output_path, index=False)
        print(f"Статистика симуляции сохранена в {output_path}")

    def generate_summary(self):
        """
        Генерирует сводный отчет по симуляции на основании собранных статистических данных.
        """
        df = self.collect_statistics()
        summary = df.describe().to_string()
        summary_path = os.path.join(self.output_dir, "simulation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Сводный отчет сохранен в {summary_path}")
        return summary

    def visualize_results(self):
        """
        Визуализирует результаты симуляции, например, изменение суммарного времени ожидания во времени.
        """
        df = self.collect_statistics()
        plt.figure(figsize=(10, 6))
        plt.plot(df["time"], df.filter(like="total_waiting").sum(axis=1), label="Общее время ожидания")
        plt.xlabel("Время (сек)")
        plt.ylabel("Время ожидания (сек)")
        plt.title("Изменение суммарного времени ожидания на перекрестках")
        plt.legend()
        vis_path = os.path.join(self.output_dir, "waiting_time_over_time.png")
        plt.savefig(vis_path)
        plt.close()
        print(f"График результатов сохранен в {vis_path}")

###############################################################################
# Класс SUMOScenarioGenerator
###############################################################################
class SUMOScenarioGenerator:
    """
    Генерирует сценарий для симуляции SUMO на базе данных OpenStreetMap.
    
    Параметры:
      - osm_file: путь к файлу OpenStreetMap (.osm)
      - output_dir: директория для сохранения сгенерированных файлов SUMO
      - network_params: параметры дорожной сети (словарь)
      - traffic_params: параметры транспортного потока (словарь)
    """
    def __init__(self, osm_file, output_dir, network_params, traffic_params):
        self.osm_file = osm_file
        self.output_dir = output_dir
        self.network_params = network_params
        self.traffic_params = traffic_params

    def generate_network(self):
        """
        Генерирует сеть SUMO из файла OSM с использованием утилиты netconvert.
        """
        network_file = os.path.join(self.output_dir, "network.net.xml")
        cmd = [
            "netconvert",
            "--osm-files", self.osm_file,
            "--output-file", network_file
        ]
        print("Генерация сети SUMO:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return network_file

    def generate_traffic_demand(self):
        """
        Генерирует спрос на транспорт, создавая файл маршрутов (routes).
        """
        routes_file = os.path.join(self.output_dir, "routes.rou.xml")
        # Простой пример: копирование предопределенного файла или генерация маршрутов с учетом входных параметров
        # Для демонстрации просто создадим пустой файл
        with open(routes_file, "w", encoding="utf-8") as f:
            f.write("<routes></routes>")
        print(f"Файл спроса на транспорт сгенерирован: {routes_file}")
        return routes_file

    def generate_config(self):
        """
        Генерирует конфигурационный файл SUMO (.sumocfg) на основе сгенерированных файлов.
        """
        net_file = self.generate_network()
        routes_file = self.generate_traffic_demand()
        config_file = os.path.join(self.output_dir, "osm.sumocfg")
        config_content = f"""<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{routes_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"Конфигурационный файл создан: {config_file}")
        return config_file

    def generate_scenario(self):
        """
        Генерирует полный сценарий для симуляции: сеть, спрос, конфигурацию.
        """
        config_file = self.generate_config()
        return config_file

    def customize_traffic_lights(self):
        """
        Настраивает параметры светофоров в сценарии. Здесь можно изменить программные настройки
        или добавить акценты для симуляции.
        """
        # Пример реализации – просто вывод сообщения.
        print("Настройка параметров светофоров завершена.")

###############################################################################
# Функция main()
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Интеграция системы умных светофоров с симулятором SUMO")
    parser.add_argument("--sumo_config", type=str, default="osm.sumocfg",
                        help="Путь к конфигурационному файлу SUMO (по умолчанию osm.sumocfg)")
    parser.add_argument("--gui", type=bool, default=True,
                        help="Использовать графический интерфейс SUMO (по умолчанию True)")
    parser.add_argument("--steps", type=int, default=3600,
                        help="Количество шагов симуляции (по умолчанию 3600)")
    parser.add_argument("--min_phase", type=int, default=10,
                        help="Минимальная продолжительность фазы для умных светофоров (по умолчанию 10)")
    parser.add_argument("--max_phase", type=int, default=60,
                        help="Максимальная продолжительность фазы для умных светофоров (по умолчанию 60)")
    parser.add_argument("--output_dir", type=str, default="./sumo_results",
                        help="Директория для сохранения результатов (по умолчанию ./sumo_results)")
    args = parser.parse_args()

    # Создаем директорию для результатов, если не существует
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Если требуется, можно сгенерировать сценарий с использованием OSM
    # Пример создания сценария:
    # scenario_generator = SUMOScenarioGenerator("map.osm", args.output_dir, network_params={}, traffic_params={})
    # sumo_config_file = scenario_generator.generate_scenario()
    # В данном примере используем переданный параметр sumo_config
    sumo_config_file = args.sumo_config

    # Инициализируем соединение с SUMO
    sumo_conn = SUMOConnection(sumo_config_file, use_gui=args.gui)
    sumo_conn.start_simulation()
    
    # Получаем список светофоров в симуляции
    tl_ids = sumo_conn.get_traffic_lights()
    print("Светофоры в симуляции:", tl_ids)
    
    # Для каждого светофора создаем детектор и контроллер
    controllers = []
    detectors = []
    optimizers = []
    # В этом примере для демонстрации используем первый светофор из списка (если он существует)
    if tl_ids:
        traffic_light_id = tl_ids[0]
        # Предполагаем, что идентификатор перекрестка можно получить из идентификатора светофора
        junction_id = traffic_light_id  # для упрощения
        detector = SUMOTrafficDetector(sumo_conn, junction_id, detector_range=50)
        detector.setup_detectors()
        detectors.append(detector)
        # Создаем контроллер – можно выбрать "traditional" или "smart"
        controller = SUMOTrafficLightController(sumo_conn, traffic_light_id, detector, controller_type="smart")
        # Если умный – создаем программу умного управления
        if controller.controller_type == "smart":
            controller.create_smart_program()
        else:
            controller.create_traditional_program()
        controllers.append(controller)
        
        # Создаем оптимизатор для умных светофоров
        optimizer = SUMOTrafficLightOptimizer(controller, detector, args.min_phase, args.max_phase, optimization_interval=50)
        optimizers.append(optimizer)
    else:
        print("Светофоры не найдены!")
        sumo_conn.stop_simulation()
        sys.exit(1)

    # Запускаем симуляцию через SUMOSimulationRunner
    simulation_runner = SUMOSimulationRunner(sumo_conn, controllers, detectors, optimizers, args.steps, args.output_dir)
    simulation_runner.run_simulation()
    simulation_runner.save_results()
    summary = simulation_runner.generate_summary()
    print("Сводный отчет симуляции:\n", summary)
    simulation_runner.visualize_results()

    # Останавливаем симуляцию
    sumo_conn.stop_simulation()

if __name__ == "__main__":
    main()
