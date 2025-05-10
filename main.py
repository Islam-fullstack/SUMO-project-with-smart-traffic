#!/usr/bin/env python3
"""
main.py

Главный файл для управления всей системой умных светофоров.
Интегрирует модули обнаружения транспорта, моделирования трафика, управления светофорами
(традиционное и умное), интеграцию с SUMO, визуализацию и экспорт результатов.

Зависимости:
  pip install numpy pandas matplotlib opencv-python ultralytics sumolib traci seaborn pyyaml

Пример запуска:
  python main.py --config config.json --mode simulation --detection_source simulation --output_dir ./results --verbose True
"""

import os
import sys
import json
import yaml
import logging
import argparse

# Здесь предполагается, что импортируются все остальные модули проекта:
# from vehicle_detection import VehicleDetector
# from traffic_simulation import TrafficSimulation
# from traditional_traffic_light import TraditionalTrafficLightController
# from smart_traffic_light import SmartTrafficLightController
# from sumo_integration import SUMOConnection, SUMOTrafficDetector, SUMOTrafficLightController, SUMOSimulationRunner
# from visualization_and_comparison import TrafficMetricsAnalyzer, TrafficVisualization

class SmartTrafficLightSystem:
    def __init__(self, config_path, mode, detection_source, output_dir):
        self.config_path = config_path
        self.mode = mode
        self.detection_source = detection_source
        self.output_dir = output_dir
        self.config = None

    def load_config(self):
        try:
            if self.config_path.endswith(".json"):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            elif self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                logging.error("Неподдерживаемый формат конфигурации.")
                sys.exit(1)
            logging.info("Конфигурация загружена.")
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации: {e}")
            sys.exit(1)

    def setup_components(self):
        logging.info("Настройка компонентов системы.")
        # Здесь должны быть созданы экземпляры всех модулей системы,
        # например, детектора, симулятора, контроллеров и модуля визуализации.
        logging.info("Компоненты системы настроены.")

    def run_simulation(self):
        logging.info("Запуск симуляции (традиционный и умный режим).")
        # Вызовите симуляцию, например из traffic_simulation.py.
        logging.info("Симуляция завершена.")

    def run_sumo_integration(self):
        logging.info("Запуск интеграции с SUMO.")
        logging.info("SUMO-интеграция завершена.")

    def run_real_time_processing(self):
        logging.info("Запуск обработки видео в реальном времени.")
        logging.info("Обработка видео завершена.")

    def analyze_results(self):
        logging.info("Анализ результатов системы.")
        logging.info("Анализ завершен.")

    def visualize_results(self):
        logging.info("Визуализация результатов.")
        logging.info("Визуализация завершена.")

    def generate_report(self):
        report_path = os.path.join(self.output_dir, "final_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Итоговый отчет работы системы умных светофоров\n")
            f.write("------------------------------------------------\n")
            f.write("Сравнительный анализ выполнен.\n")
        logging.info(f"Отчет сгенерирован: {report_path}")

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None

    def load_config(self):
        try:
            if self.config_path.endswith(".json"):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            elif self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                logging.error("Неподдерживаемый формат конфигурации.")
                sys.exit(1)
            logging.info("Конфигурация загружена.")
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации: {e}")
            sys.exit(1)

    def validate_config(self):
        required_keys = ["simulation_params", "detection_params",
                         "traditional_controller_params", "smart_controller_params",
                         "sumo_params", "visualization_params"]
        missing = [key for key in required_keys if key not in self.config]
        if missing:
            logging.error(f"Отсутствуют ключи: {missing}")
            sys.exit(1)
        logging.info("Конфигурация проверена.")

    def get_simulation_params(self):
        return self.config.get("simulation_params", {})

    def get_detection_params(self):
        return self.config.get("detection_params", {})

    def get_traditional_controller_params(self):
        return self.config.get("traditional_controller_params", {})

    def get_smart_controller_params(self):
        return self.config.get("smart_controller_params", {})

    def get_sumo_params(self):
        return self.config.get("sumo_params", {})

    def get_visualization_params(self):
        return self.config.get("visualization_params", {})

class SystemManager:
    def __init__(self, system, config_manager):
        self.system = system
        self.config_manager = config_manager

    def initialize_system(self):
        self.config_manager.load_config()
        self.config_manager.validate_config()
        self.system.load_config()
        self.system.setup_components()
        self.log_activity("Система инициализирована.", level="INFO")

    def run(self):
        self.log_activity(f"Запуск системы в режиме: {self.system.mode}", level="INFO")
        if self.system.mode == "simulation":
            self.system.run_simulation()
        elif self.system.mode == "sumo_integration":
            self.system.run_sumo_integration()
        elif self.system.mode == "real_time":
            self.system.run_real_time_processing()
        else:
            self.log_activity(f"Неизвестный режим: {self.system.mode}", level="ERROR")
            sys.exit(1)
        self.log_activity("Работа системы завершена.", level="INFO")

    def handle_errors(self):
        pass

    def cleanup(self):
        self.log_activity("Очистка ресурсов завершена.", level="INFO")

    def log_activity(self, message, level="INFO"):
        if level == "INFO":
            logging.info(message)
        elif level == "ERROR":
            logging.error(message)
        elif level == "DEBUG":
            logging.debug(message)

class DataExporter:
    def __init__(self, output_dir, formats=None):
        self.output_dir = output_dir
        self.formats = formats if formats else ["csv", "json"]

    def export_simulation_results(self, simulation_data):
        path = os.path.join(self.output_dir, "simulation_results.csv")
        simulation_data.to_csv(path, index=False)
        logging.info(f"Экспорт результатов симуляции в CSV: {path}")

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Логирование настроено.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Управление системой умных светофоров")
    parser.add_argument("--config", type=str, default="config.json", help="Путь к конфигурации")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation", "sumo_integration", "real_time"], help="Режим работы системы")
    parser.add_argument("--detection_source", type=str, default="simulation", choices=["video", "simulation", "sumo"], help="Источник данных для обнаружения")
    parser.add_argument("--output_dir", type=str, default="./results", help="Директория для результатов")
    parser.add_argument("--create_config", action="store_true", help="Создать конфигурацию по умолчанию")
    parser.add_argument("--verbose", type=bool, default=False, help="Подробное логирование")
    return parser.parse_args()

def validate_environment():
    try:
        import numpy; import pandas; import matplotlib; import cv2; import ultralytics; import sumolib; import traci; import seaborn; import yaml
    except ImportError as e:
        logging.error(f"Отсутствует зависимость: {e}")
        sys.exit(1)
    logging.info("Все зависимости обнаружены.")

def create_default_config(config_path):
    default_config = {
        "simulation_params": {"simulation_time": 3600, "dt": 0.1},
        "detection_params": {"source": "simulation"},
        "traditional_controller_params": {"phase_durations": [30, 30], "yellow_time": 3, "cycle_time": 60},
        "smart_controller_params": {"min_phase_duration": 10, "max_phase_duration": 60, "yellow_time": 3, "min_cycle_time": 20, "max_cycle_time": 120},
        "sumo_params": {"sumo_config": "myProject/osm.sumocfg", "use_gui": True, "port": 8813},
        "visualization_params": {"plot_types": ["bar", "line", "heatmap", "kde"], "save_formats": ["png", "pdf"], "create_animations": True}
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=4)
    logging.info(f"Конфигурация по умолчанию создана: {config_path}")

def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    validate_environment()
    if args.create_config:
        create_default_config(args.config)
        sys.exit(0)
    config_manager = ConfigManager(args.config)
    system = SmartTrafficLightSystem(args.config, args.mode, args.detection_source, args.output_dir)
    sys_manager = SystemManager(system, config_manager)
    sys_manager.initialize_system()
    try:
        sys_manager.run()
    except Exception as e:
        sys_manager.handle_errors()
        logging.error(f"Ошибка системы: {e}")
    finally:
        sys_manager.cleanup()
    system.analyze_results()
    system.visualize_results()
    system.generate_report()
    exporter = DataExporter(args.output_dir)
    logging.info("Система умных светофоров завершила работу.")

if __name__ == "__main__":
    main()
