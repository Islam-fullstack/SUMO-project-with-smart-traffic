#!/usr/bin/env python3
"""
sumo_integration.py

Интеграция системы умных светофоров с симулятором SUMO.
Файлы SUMO находятся в каталоге myProject.

Зависимости:
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
import traci
import sumolib

class SUMOConnection:
    def __init__(self, sumo_config_path, use_gui=True, port=8813):
        self.sumo_config_path = sumo_config_path
        self.use_gui = use_gui
        self.port = port
        self.sumo_process = None

    def start_simulation(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_config_path, "--remote-port", str(self.port)]
        print("Запуск SUMO с командой:", " ".join(sumo_cmd))
        self.sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        traci.init(port=self.port)
        print("SUMO запущен и TraCI подключен.")

    def stop_simulation(self):
        traci.close()
        if self.sumo_process:
            self.sumo_process.terminate()
            self.sumo_process.wait()
        print("SUMO остановлен.")

    def get_step(self):
        return traci.simulation.getTime()

    def step(self):
        traci.simulationStep()

    def get_traffic_lights(self):
        return traci.trafficlight.getIDList()

class SUMOTrafficDetector:
    def __init__(self, sumo_connection, junction_id, detector_range):
        self.sumo_connection = sumo_connection
        self.junction_id = junction_id
        self.detector_range = detector_range
        self.incoming_lanes = []
        self.detection_data = {}

    def setup_detectors(self):
        all_lanes = traci.lane.getIDList()
        self.incoming_lanes = [lane for lane in all_lanes if self.junction_id in lane]
        print(f"Для перекрестка {self.junction_id} определены детекторы для полос: {self.incoming_lanes}")

    def get_vehicle_counts(self):
        counts = {}
        for lane in self.incoming_lanes:
            counts[lane] = traci.lane.getLastStepVehicleNumber(lane)
        return counts

    def get_waiting_times(self):
        waiting = {}
        for lane in self.incoming_lanes:
            waiting[lane] = traci.lane.getWaitingTime(lane)
        return waiting

    def get_queue_lengths(self):
        return self.get_vehicle_counts()

    def get_traffic_density(self):
        density = {}
        for lane in self.incoming_lanes:
            veh = traci.lane.getLastStepVehicleNumber(lane)
            length = traci.lane.getLength(lane)
            density[lane] = veh / length if length > 0 else 0
        return density

    def update(self):
        self.detection_data = {
            "vehicle_counts": self.get_vehicle_counts(),
            "waiting_times": self.get_waiting_times(),
            "queue_lengths": self.get_queue_lengths(),
            "traffic_density": self.get_traffic_density()
        }
        print(f"[DEBUG] Детектор {self.junction_id}: {self.detection_data}")

    def export_detection_data(self, output_path):
        df = pd.DataFrame(self.detection_data)
        df.to_csv(output_path, index=False)
        print(f"Данные детектора экспортированы в {output_path}")

class SUMOTrafficLightController:
    def __init__(self, sumo_connection, traffic_light_id, detector, controller_type="smart"):
        self.sumo_connection = sumo_connection
        self.traffic_light_id = traffic_light_id
        self.detector = detector
        self.controller_type = controller_type

    def get_current_phase(self):
        return traci.trafficlight.getPhase(self.traffic_light_id)

    def set_phase(self, phase_index):
        traci.trafficlight.setPhase(self.traffic_light_id, phase_index)
        print(f"Светофор {self.traffic_light_id}: установлена фаза {phase_index}")

    def set_phase_duration(self, phase_index, duration):
        traci.trafficlight.setPhaseDuration(self.traffic_light_id, duration)
        print(f"Светофор {self.traffic_light_id}: длительность фазы {phase_index} установлена в {duration} сек")

    def get_program(self):
        return traci.trafficlight.getCompleteRedYellowGreenDefinition(self.traffic_light_id)

    def set_program(self, program_id):
        traci.trafficlight.setProgram(self.traffic_light_id, program_id)
        print(f"Светофор {self.traffic_light_id}: установлена программа {program_id}")

    def create_smart_program(self):
        current_program = self.get_program()
        print(f"Светофор {self.traffic_light_id}: сохраняем текущую программу для умного управления: {current_program}")

    def update(self, step):
        if self.controller_type == "smart":
            self.detector.update()
            waiting = self.detector.get_waiting_times()
            if sum(waiting.values()) > 20:
                current_phase = self.get_current_phase()
                next_phase = (current_phase + 1) % 4
                self.set_phase(next_phase)
        else:
            cycle_time = 30
            if step % cycle_time == 0:
                current_phase = self.get_current_phase()
                next_phase = (current_phase + 1) % 4
                self.set_phase(next_phase)

class SUMOTrafficLightOptimizer:
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

    def update(self, step):
        if step - self.last_optimization_step >= self.optimization_interval:
            self.optimize_phase_durations()
            self.last_optimization_step = step

class SUMOSimulationRunner:
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
        if df.empty:
            print("[WARN] Статистика симуляции пустая!")
        return df

    def save_results(self):
        df = self.collect_statistics()
        output_path = os.path.join(self.output_dir, "simulation_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Статистика симуляции сохранена в {output_path}")

    def generate_summary(self):
        df = self.collect_statistics()
        summary = df.describe().to_string() if not df.empty else "Нет данных"
        summary_path = os.path.join(self.output_dir, "simulation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Сводный отчет сохранен в {summary_path}")
        return summary

    def visualize_results(self):
        df = self.collect_statistics()
        if df.empty:
            print("[WARN] Нет данных для визуализации.")
            return
        plt.figure(figsize=(10, 6))
        waiting_cols = [col for col in df.columns if "total_waiting" in col]
        if waiting_cols:
            plt.plot(df["time"], df[waiting_cols].sum(axis=1), label="Общее время ожидания")
            plt.xlabel("Время (сек)")
            plt.ylabel("Время ожидания (сек)")
            plt.title("Изменение суммарного времени ожидания на перекрестках")
            plt.legend()
            vis_path = os.path.join(self.output_dir, "waiting_time_over_time.png")
            plt.savefig(vis_path)
            plt.close()
            print(f"График результатов сохранен в {vis_path}")
        else:
            print("[WARN] Колонки с данными времени ожидания не найдены.")

def main():
    parser = argparse.ArgumentParser(description="Интеграция системы умных светофоров с SUMO")
    parser.add_argument("--sumo_config", type=str, default="myProject/osm.sumocfg",
                        help="Путь к файлу SUMO-конфигурации (по умолчанию: myProject/osm.sumocfg)")
    parser.add_argument("--gui", type=bool, default=True, help="Использовать графический интерфейс SUMO")
    parser.add_argument("--steps", type=int, default=3600, help="Количество шагов симуляции")
    parser.add_argument("--min_phase", type=int, default=10, help="Минимальная длительность фазы для умного светофора")
    parser.add_argument("--max_phase", type=int, default=60, help="Максимальная длительность фазы для умного светофора")
    parser.add_argument("--output_dir", type=str, default="./results", help="Директория для результатов")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sumo_conn = SUMOConnection(args.sumo_config, use_gui=args.gui)
    sumo_conn.start_simulation()
    tl_ids = sumo_conn.get_traffic_lights()
    print("Светофоры в симуляции:", tl_ids)
    if not tl_ids:
        print("Светофоры не найдены!")
        sumo_conn.stop_simulation()
        sys.exit(1)
    traffic_light_id = tl_ids[0]
    junction_id = traffic_light_id
    detector = SUMOTrafficDetector(sumo_conn, junction_id, detector_range=50)
    detector.setup_detectors()
    controller = SUMOTrafficLightController(sumo_conn, traffic_light_id, detector, controller_type="smart")
    controller.create_smart_program()
    optimizer = SUMOTrafficLightOptimizer(controller, detector, args.min_phase, args.max_phase, optimization_interval=50)
    runner = SUMOSimulationRunner(sumo_conn, [controller], [detector], [optimizer], args.steps, args.output_dir)
    runner.run_simulation()
    runner.save_results()
    summary = runner.generate_summary()
    print("Сводный отчет симуляции:\n", summary)
    runner.visualize_results()
    sumo_conn.stop_simulation()

if __name__ == "__main__":
    main()
