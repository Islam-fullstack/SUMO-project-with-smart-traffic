#!/usr/bin/env python3
"""
sumo_integration.py

Интеграция разработанной системы умных светофоров с микроскопическим симулятором
дорожного движения SUMO (Simulation of Urban MObility). Система использует данные OSM
и работает с перекрестками, которые уже созданы в проекте SUMO.

Перед использованием убедитесь, что все SUMO файлы (osm.sumocfg, osm.net.xml.gz, 
osm.passenger.trips.xml, osm.poly.xml.gz и т.д.) находятся в каталоге myProject.

Зависимости:
  - sumolib
  - traci
  - numpy
  - pandas
  - matplotlib

Установка зависимостей:
  pip install sumolib traci numpy pandas matplotlib

Пример запуска:
  python sumo_integration.py
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
        Запускает симуляцию SUMO с использованием TraCI.
        Если use_gui=True, используется sumo-gui, иначе – sumo.
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
        """
        all_lanes = traci.lane.getIDList()
        # Выбираем те полосы, у которых в ID фигурирует идентификатор перекрестка
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
        return self.get_vehicle_counts()

    def get_traffic_density(self):
        """
        Возвращает плотность транспортного потока для каждой входящей полосы.
        """
        density = {}
        for lane in self.incoming_lanes:
            veh_num = traci.lane.getLastStepVehicleNumber(lane)
            lane_length = traci.lane.getLength(lane)
            density[lane] = veh_num / lane_length if lane_length > 0 else 0
        return density

    def update(self):
        """
        Обновляет данные детектора.
        """
        self.detection_data = {
            "vehicle_counts": self.get_vehicle_counts(),
            "vehicle_speeds": self.get_vehicle_speeds(),
            "waiting_times": self.get_waiting_times(),
            "queue_lengths": self.get_queue_lengths(),
            "traffic_density": self.get_traffic_density()
        }

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
        """
        return traci.trafficlight.getNextSwitch(self.traffic_light_id)

    def set_phase_duration(self, phase_index, duration):
        """
        Устанавливает продолжительность указанной фазы.
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
        """
        # Допустим, мы используем существующую программу из SUMO, поэтому просто выводим сообщение.
        current_program = self.get_program()
        print(f"Светофор {self.traffic_light_id}: используется традиционная программа: {current_program}")

    def create_smart_program(self):
        """
        Создает умную программу с адаптивными фазами.
        
        Важно: вместо того, чтобы переключать светофор на неизвестную программу (например, 'actuated' или 'static'),
        мы оставляем текущую программу. Это позволяет избежать ошибки, если в SUMO нет такой программы.
        """
        current_program = self.get_program()
        print(f"Светофор {self.traffic_light_id}: используется текущая программа для умного управления: {current_program}")
        # Если необходимо, можно дополнительно изменять длительность фаз с помощью set_phase_duration.

    def update(self, step):
        """
        Обновляет состояние контроллера на основе текущего шага симуляции и данных детектора.
        Для умного режима переключаем фазу, если суммарное время ожидания превышает порог.
        Для традиционного режима переключаем фазу циклически.
        """
        if self.controller_type == "smart":
            self.detector.update()
            waiting = self.detector.get_waiting_times()
            total_wait = sum(waiting.values())
            if total_wait > 20:
                current_phase = self.get_current_phase()
                next_phase = (current_phase + 1) % 4  # предположим, что есть 4 фазы
                self.set_phase(next_phase)
        else:
            cycle_time = 30  # пример фиксированного цикла
            if step % cycle_time == 0:
                current_phase = self.get_current_phase()
                next_phase = (current_phase + 1) % 4
                self.set_phase(next_phase)

###############################################################################
# Класс SUMOTrafficLightOptimizer
###############################################################################
class SUMOTrafficLightOptimizer:
    """
    Оптимизатор для умных светофоров. Оптимизирует продолжительность фаз и выбирает следующую фазу.
    
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
        self.detector.update()
        counts = self.detector.get_vehicle_counts()
        waiting = self.detector.get_waiting_times()
        score = sum(counts.values()) + 0.5 * sum(waiting.values())
        print(f"Оценка фазы {phase_index}: {score}")
        return score

    def optimize_phase_durations(self):
        score = self.calculate_phase_score(self.controller.get_current_phase())
        normalized = min(score / 50.0, 1.0)
        new_duration = self.min_phase_duration + (self.max_phase_duration - self.min_phase_duration) * normalized
        current_phase = self.controller.get_current_phase()
        self.controller.set_phase_duration(current_phase, new_duration)
        print(f"Оптимизированная длительность фазы {current_phase}: {new_duration} сек")

    def select_next_phase(self):
        scores = []
        for phase in range(4):  # предположим 4 фазы
            score = self.calculate_phase_score(phase)
            scores.append(score)
        next_phase = np.argmax(scores)
        print(f"Выбрана следующая фаза: {next_phase}")
        return next_phase

    def update(self, step):
        if step - self.last_optimization_step >= self.optimization_interval:
            self.optimize_phase_durations()
            self.last_optimization_step = step

    def apply_optimization(self):
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
        self.statistics = []

    def run_simulation(self):
        for step in range(self.simulation_steps):
            self.sumo_connection.step()
            current_time = self.sumo_connection.get_step()
            for controller in self.controllers:
                controller.update(step)
            for optimizer in self.optimizers:
                optimizer.update(step)
            for detector in self.detectors:
                detector.update()
            stats = {"step": step, "time": current_time}
            for detector in self.detectors:
                waiting = detector.get_waiting_times()
                stats[f"{detector.junction_id}_total_waiting"] = sum(waiting.values())
            self.statistics.append(stats)
        print("Симуляция завершена.")

    def collect_statistics(self):
        df = pd.DataFrame(self.statistics)
        return df

    def save_results(self):
        df = self.collect_statistics()
        output_path = os.path.join(self.output_dir, "simulation_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Статистика симуляции сохранена в {output_path}")

    def generate_summary(self):
        df = self.collect_statistics()
        summary = df.describe().to_string()
        summary_path = os.path.join(self.output_dir, "simulation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Сводный отчет сохранен в {summary_path}")
        return summary

    def visualize_results(self):
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
    Генерирует сценарий для симуляции SUMO на базе OSM.
    
    Параметры:
      - osm_file: путь к файлу OpenStreetMap (.osm)
      - output_dir: директория для сохранения файлов SUMO
      - network_params: параметры сети
      - traffic_params: параметры транспортного потока
    """
    def __init__(self, osm_file, output_dir, network_params, traffic_params):
        self.osm_file = osm_file
        self.output_dir = output_dir
        self.network_params = network_params
        self.traffic_params = traffic_params

    def generate_network(self):
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
        routes_file = os.path.join(self.output_dir, "routes.rou.xml")
        with open(routes_file, "w", encoding="utf-8") as f:
            f.write("<routes></routes>")
        print(f"Файл спроса на транспорт сгенерирован: {routes_file}")
        return routes_file

    def generate_config(self):
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
        config_file = self.generate_config()
        return config_file

    def customize_traffic_lights(self):
        print("Настройка параметров светофоров завершена.")

###############################################################################
# Функция main()
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Интеграция системы умных светофоров с SUMO")
    parser.add_argument("--sumo_config", type=str, default="myProject/osm.sumocfg",
                        help="Путь к конфигурационному файлу SUMO (по умолчанию myProject/osm.sumocfg)")
    parser.add_argument("--gui", type=bool, default=True,
                        help="Использовать графический интерфейс SUMO (по умолчанию True)")
    parser.add_argument("--steps", type=int, default=3600,
                        help="Количество шагов симуляции (по умолчанию 3600)")
    parser.add_argument("--min_phase", type=int, default=10,
                        help="Минимальная продолжительность фазы для умных светофоров (по умолчанию 10)")
    parser.add_argument("--max_phase", type=int, default=60,
                        help="Максимальная продолжительность фазы для умных светофоров (по умолчанию 60)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Директория для сохранения результатов (по умолчанию ./results)")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Запуск SUMO
    sumo_conn = SUMOConnection(args.sumo_config, use_gui=args.gui, port=8813)
    sumo_conn.start_simulation()

    # Получаем список светофоров
    tl_ids = sumo_conn.get_traffic_lights()
    print("Светофоры в симуляции:", tl_ids)

    # Для демонстрации используем первый светофор (например, '3455267510')
    if not tl_ids:
        print("Светофоры не найдены!")
        sumo_conn.stop_simulation()
        sys.exit(1)

    traffic_light_id = tl_ids[0]
    junction_id = traffic_light_id  # Для упрощения считаем, что ID светофора совпадает с ID перекрестка

    # Создаем детектор для перекрестка
    detector = SUMOTrafficDetector(sumo_conn, junction_id, detector_range=50)
    detector.setup_detectors()

    # Создаем контроллер (умный)
    controller = SUMOTrafficLightController(sumo_conn, traffic_light_id, detector, controller_type="smart")
    controller.create_smart_program()  # теперь не будет пытаться переключать на неизвестную программу

    # Создаем оптимизатор для умного контроллера
    optimizer = SUMOTrafficLightOptimizer(controller, detector, args.min_phase, args.max_phase, optimization_interval=50)

    # Запускаем симуляцию через SUMOSimulationRunner
    simulation_runner = SUMOSimulationRunner(sumo_conn, [controller], [detector], [optimizer], args.steps, args.output_dir)
    simulation_runner.run_simulation()
    simulation_runner.save_results()
    summary = simulation_runner.generate_summary()
    print("Сводный отчет симуляции:\n", summary)
    simulation_runner.visualize_results()

    sumo_conn.stop_simulation()

if __name__ == "__main__":
    main()
