#!/usr/bin/env python3
"""
visualization_and_comparison.py

Скрипт для визуализации и сравнения результатов моделирования традиционного 
и умного алгоритмов управления светофорами. Для создания наглядных графиков и диаграмм 
используются библиотеки:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy (для статистического анализа)
  - ffmpeg (для сохранения анимаций, опционально)

Установка зависимостей:
  pip install numpy pandas matplotlib seaborn scipy

Пример запуска из командной строки:
  python visualization_and_comparison.py --traditional_results traditional_results.csv \
       --smart_results smart_results.csv --output_dir ./comparison_results \
       --create_animations --detailed_analysis

Автор: Islam
Дата: 2025-05-09
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from matplotlib.animation import FuncAnimation, FFMpegWriter

###############################################################################
# Класс TrafficMetricsAnalyzer
###############################################################################
class TrafficMetricsAnalyzer:
    """
    Анализатор транспортных метрик. Загружает результаты моделирования традиционного и 
    умного алгоритмов, рассчитывает ключевые показатели эффективности и генерирует сводный 
    текстовый отчет.

    Параметры:
      - traditional_results_path: путь к файлу с результатами традиционного алгоритма.
      - smart_results_path: путь к файлу с результатами умного алгоритма.
      - output_dir: директория для сохранения графиков и отчетов.
    """
    def __init__(self, traditional_results_path, smart_results_path, output_dir):
        self.traditional_results_path = traditional_results_path
        self.smart_results_path = smart_results_path
        self.output_dir = output_dir
        self.traditional_df = None
        self.smart_df = None
        self.traditional_metrics = {}
        self.smart_metrics = {}
        
    def load_data(self):
        """Загружает данные из CSV-файлов и сохраняет их в атрибуты DataFrame."""
        self.traditional_df = pd.read_csv(self.traditional_results_path)
        self.smart_df = pd.read_csv(self.smart_results_path)
        print("Данные загружены.")

    def calculate_metrics(self):
        """
        Рассчитывает ключевые метрики:
          - Среднее время ожидания (по столбцам *_waiting)
          - Средняя длина очереди (по столбцам *_queue)
          - Пропускная способность, оценённая по разностям очередей
          - Дополнительно можно оценить выбросы CO2 и расход топлива на основе пропускной способности
        """
        def compute_avg_metric(df, metric_suffix):
            cols = [col for col in df.columns if col.endswith(metric_suffix)]
            if not cols:
                return None
            return df[cols].mean().mean()
        
        def compute_throughput(df):
            # Расчет департур как суммарное снижение очереди между конмерами, усреднённое по времени.
            throughput_vals = {}
            queue_cols = [col for col in df.columns if col.endswith("_queue")]
            for col in queue_cols:
                diffs = df[col].diff().fillna(0)
                # Если очередь уменьшилась, считаем эти изменения как "отработка" (departures)
                departures = diffs[diffs < 0].abs().sum()
                throughput_vals[col] = departures / (df['time'].iloc[-1] - df['time'].iloc[0])
            if throughput_vals:
                return np.mean(list(throughput_vals.values()))
            return None

        self.traditional_metrics['avg_waiting_time'] = compute_avg_metric(self.traditional_df, "_waiting")
        self.smart_metrics['avg_waiting_time'] = compute_avg_metric(self.smart_df, "_waiting")
        
        self.traditional_metrics['avg_queue_length'] = compute_avg_metric(self.traditional_df, "_queue")
        self.smart_metrics['avg_queue_length'] = compute_avg_metric(self.smart_df, "_queue")
        
        self.traditional_metrics['throughput'] = compute_throughput(self.traditional_df)
        self.smart_metrics['throughput'] = compute_throughput(self.smart_df)
        
        # Optionally, estimate emissions and fuel consumption
        # Например, пусть выбросы CO2 пропорциональны (throughput * 100) и расход топлива = throughput * 10
        self.traditional_metrics['estimated_emissions'] = self.traditional_metrics['throughput'] * 100 if self.traditional_metrics['throughput'] is not None else None
        self.smart_metrics['estimated_emissions'] = self.smart_metrics['throughput'] * 100 if self.smart_metrics['throughput'] is not None else None
        
        self.traditional_metrics['fuel_consumption'] = self.traditional_metrics['throughput'] * 10 if self.traditional_metrics['throughput'] is not None else None
        self.smart_metrics['fuel_consumption'] = self.smart_metrics['throughput'] * 10 if self.smart_metrics['throughput'] is not None else None
        
        print("Метрики рассчитаны.")
    
    def compare_waiting_times(self):
        """Сравнивает среднее время ожидания между традиционным и умным алгоритмами."""
        trad = self.traditional_metrics.get('avg_waiting_time', np.nan)
        smart = self.smart_metrics.get('avg_waiting_time', np.nan)
        return {'traditional': trad, 'smart': smart}
    
    def compare_queue_lengths(self):
        """Сравнивает средние длины очередей между традиционным и умным алгоритмами."""
        trad = self.traditional_metrics.get('avg_queue_length', np.nan)
        smart = self.smart_metrics.get('avg_queue_length', np.nan)
        return {'traditional': trad, 'smart': smart}
    
    def compare_throughput(self):
        """Сравнивает пропускную способность между традиционным и умным алгоритмами."""
        trad = self.traditional_metrics.get('throughput', np.nan)
        smart = self.smart_metrics.get('throughput', np.nan)
        return {'traditional': trad, 'smart': smart}
    
    def compare_emissions(self):
        """Опционально сравнивает оценочные выбросы CO2."""
        trad = self.traditional_metrics.get('estimated_emissions', np.nan)
        smart = self.smart_metrics.get('estimated_emissions', np.nan)
        return {'traditional': trad, 'smart': smart}
    
    def compare_fuel_consumption(self):
        """Опционально сравнивает оценочный расход топлива."""
        trad = self.traditional_metrics.get('fuel_consumption', np.nan)
        smart = self.smart_metrics.get('fuel_consumption', np.nan)
        return {'traditional': trad, 'smart': smart}
    
    def generate_summary_report(self):
        """Генерирует текстовый отчет с ключевыми выводами и сохраняет его в output_dir."""
        report_lines = [
            "Сводный отчет по моделированию работы светофоров",
            "--------------------------------------------------------",
            f"Среднее время ожидания (сек):",
            f"  Традиционный алгоритм: {self.traditional_metrics.get('avg_waiting_time', 'N/A'):.2f}",
            f"  Умный алгоритм:        {self.smart_metrics.get('avg_waiting_time', 'N/A'):.2f}",
            "",
            f"Средняя длина очереди (единиц):",
            f"  Традиционный алгоритм: {self.traditional_metrics.get('avg_queue_length', 'N/A'):.2f}",
            f"  Умный алгоритм:        {self.smart_metrics.get('avg_queue_length', 'N/A'):.2f}",
            "",
            f"Пропускная способность (ед/сек):",
            f"  Традиционный алгоритм: {self.traditional_metrics.get('throughput', 'N/A'):.2f}",
            f"  Умный алгоритм:        {self.smart_metrics.get('throughput', 'N/A'):.2f}",
            "",
            f"Оценочные выбросы CO2 (ед):",
            f"  Традиционный алгоритм: {self.traditional_metrics.get('estimated_emissions', 'N/A'):.2f}",
            f"  Умный алгоритм:        {self.smart_metrics.get('estimated_emissions', 'N/A'):.2f}",
            "",
            f"Расход топлива (ед):",
            f"  Традиционный алгоритм: {self.traditional_metrics.get('fuel_consumption', 'N/A'):.2f}",
            f"  Умный алгоритм:        {self.smart_metrics.get('fuel_consumption', 'N/A'):.2f}",
        ]
        report = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, "summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Отчет сохранен в {report_path}")
        return report

###############################################################################
# Класс TrafficVisualization
###############################################################################
class TrafficVisualization:
    """
    Класс для визуализации метрик. Использует данные из TrafficMetricsAnalyzer и
    создает различные графики, сохраняя их в указанной директории.

    Параметры:
      - metrics_analyzer: экземпляр TrafficMetricsAnalyzer.
      - output_dir: директория для сохранения визуализаций.
    """
    def __init__(self, metrics_analyzer, output_dir):
        self.analyzer = metrics_analyzer
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_waiting_time_comparison(self):
        """Создает график сравнения среднего времени ожидания."""
        comp = self.analyzer.compare_waiting_times()
        labels = list(comp.keys())
        values = list(comp.values())
        plt.figure(figsize=(6,4))
        sns.barplot(x=labels, y=values, palette="viridis")
        plt.title("Сравнение среднего времени ожидания")
        plt.ylabel("Время ожидания (сек)")
        plt.xlabel("Алгоритм")
        fig_path = os.path.join(self.output_dir, "waiting_time_comparison.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"График времени ожидания сохранен: {fig_path}")

    def plot_queue_length_comparison(self):
        """Создает график сравнения средней длины очередей."""
        comp = self.analyzer.compare_queue_lengths()
        labels = list(comp.keys())
        values = list(comp.values())
        plt.figure(figsize=(6,4))
        sns.barplot(x=labels, y=values, palette="magma")
        plt.title("Сравнение средней длины очередей")
        plt.ylabel("Длина очереди (единиц)")
        plt.xlabel("Алгоритм")
        fig_path = os.path.join(self.output_dir, "queue_length_comparison.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"График длины очереди сохранен: {fig_path}")

    def plot_throughput_comparison(self):
        """Создает график сравнения пропускной способности."""
        comp = self.analyzer.compare_throughput()
        labels = list(comp.keys())
        values = list(comp.values())
        plt.figure(figsize=(6,4))
        sns.barplot(x=labels, y=values, palette="cubehelix")
        plt.title("Сравнение пропускной способности")
        plt.ylabel("Пропускная способность (ед/сек)")
        plt.xlabel("Алгоритм")
        fig_path = os.path.join(self.output_dir, "throughput_comparison.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"График пропускной способности сохранен: {fig_path}")

    def plot_waiting_time_by_direction(self):
        """
        Создает график распределения времени ожидания по направлениям для обоих алгоритмов.
        Предполагается, что столбцы имеют вид I1_<direction>_waiting.
        """
        def aggregate_by_direction(df):
            # Извлекаем столбцы, группируем по направлению
            waiting_cols = [col for col in df.columns if col.endswith("_waiting")]
            data = {}
            for col in waiting_cols:
                # Предполагаем, что имя столбца имеет формат "I1_<direction>_waiting"
                parts = col.split("_")
                if len(parts) >= 3:
                    direction = parts[1]
                    data.setdefault(direction, []).extend(df[col].values)
            return data
        
        trad_data = aggregate_by_direction(self.analyzer.traditional_df)
        smart_data = aggregate_by_direction(self.analyzer.smart_df)
        
        directions = sorted(list(set(list(trad_data.keys()) + list(smart_data.keys()))))
        plt.figure(figsize=(10,6))
        for d in directions:
            trad = trad_data.get(d, [])
            smart = smart_data.get(d, [])
            sns.kdeplot(trad, label=f"Традиционный {d}", shade=True)
            sns.kdeplot(smart, label=f"Умный {d}", shade=True)
        plt.title("Распределение времени ожидания по направлениям")
        plt.xlabel("Время ожидания (сек)")
        plt.ylabel("Плотность")
        fig_path = os.path.join(self.output_dir, "waiting_time_by_direction.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"График распределения времени ожидания по направлениям сохранен: {fig_path}")

    def plot_queue_length_by_direction(self):
        """
        Создает график распределения длины очередей по направлениям.
        Предполагается, что столбцы имеют вид I1_<direction>_queue.
        """
        def aggregate_by_direction(df):
            queue_cols = [col for col in df.columns if col.endswith("_queue")]
            data = {}
            for col in queue_cols:
                parts = col.split("_")
                if len(parts) >= 3:
                    direction = parts[1]
                    data.setdefault(direction, []).extend(df[col].values)
            return data
        
        trad_data = aggregate_by_direction(self.analyzer.traditional_df)
        smart_data = aggregate_by_direction(self.analyzer.smart_df)
        
        directions = sorted(list(set(list(trad_data.keys()) + list(smart_data.keys()))))
        plt.figure(figsize=(10,6))
        for d in directions:
            trad = trad_data.get(d, [])
            smart = smart_data.get(d, [])
            sns.kdeplot(trad, label=f"Традиционный {d}", shade=True)
            sns.kdeplot(smart, label=f"Умный {d}", shade=True)
        plt.title("Распределение длины очередей по направлениям")
        plt.xlabel("Длина очереди (единиц)")
        plt.ylabel("Плотность")
        fig_path = os.path.join(self.output_dir, "queue_length_by_direction.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"График распределения длины очередей сохранен: {fig_path}")

    def plot_traffic_density_heatmap(self):
        """
        Создает тепловую карту плотности трафика по направлению и времени.
        Здесь плотность вычисляется как отношение очереди к длине полосы, если такие данные присутствуют.
        """
        # Для демонстрации используем умные данные и рассчитываем плотность по времени,
        # предполагая, что каждая строка – снимок в момент времени.
        density_cols = [col for col in self.analyzer.smart_df.columns if col.endswith("_queue")]
        if not density_cols:
            print("Нет данных для построения тепловой карты плотности.")
            return
        # Для простоты переименуем столбцы, оставив направление
        density_data = self.analyzer.smart_df[density_cols].copy()
        new_cols = {}
        for col in density_data.columns:
            parts = col.split("_")
            if len(parts) >= 3:
                new_cols[col] = parts[1]
            else:
                new_cols[col] = col
        density_data.rename(columns=new_cols, inplace=True)
        # Построим тепловую карту: строки - время, столбцы - направления
        plt.figure(figsize=(8,6))
        sns.heatmap(density_data.T, cmap="YlGnBu")
        plt.title("Тепловая карта плотности трафика (очереди)")
        plt.xlabel("Время (индекс шага)")
        plt.ylabel("Направление")
        fig_path = os.path.join(self.output_dir, "traffic_density_heatmap.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Тепловая карта плотности трафика сохранена: {fig_path}")

    def plot_phase_duration_distribution(self):
        """
        Создает гистограмму распределения продолжительности фаз для умного алгоритма.
        Если в данных умного алгоритма присутствует столбец 'phase_duration', строит гистограмму.
        """
        if "phase_duration" not in self.analyzer.smart_df.columns:
            print("Данных о продолжительности фаз не обнаружено.")
            return
        plt.figure(figsize=(6,4))
        sns.histplot(self.analyzer.smart_df["phase_duration"], bins=20, kde=True)
        plt.title("Распределение продолжительности фаз (умный алгоритм)")
        plt.xlabel("Продолжительность фазы (сек)")
        fig_path = os.path.join(self.output_dir, "phase_duration_distribution.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Гистограмма распределения продолжительности фаз сохранена: {fig_path}")

    def plot_metrics_over_time(self):
        """
        Строит график изменений ключевых метрик (например, среднего времени ожидания, очереди, пропускной способности) во времени.
        Предполагается, что столбец 'time' присутствует в данных.
        """
        df_trad = self.analyzer.traditional_df.copy()
        df_smart = self.analyzer.smart_df.copy()
        
        plt.figure(figsize=(12,8))
        if "time" in df_trad.columns:
            # Вычисляем средние значения по всем колонкам с окончанием '_waiting'
            trad_wait_cols = [col for col in df_trad.columns if col.endswith("_waiting")]
            smart_wait_cols = [col for col in df_smart.columns if col.endswith("_waiting")]
            
            df_trad['avg_waiting'] = df_trad[trad_wait_cols].mean(axis=1)
            df_smart['avg_waiting'] = df_smart[smart_wait_cols].mean(axis=1)
            
            plt.plot(df_trad["time"], df_trad['avg_waiting'], label="Традиционный: Время ожидания", color="blue")
            plt.plot(df_smart["time"], df_smart['avg_waiting'], label="Умный: Время ожидания", color="green")
        plt.title("Изменение среднего времени ожидания во времени")
        plt.xlabel("Время (сек)")
        plt.ylabel("Время ожидания (сек)")
        fig_path = os.path.join(self.output_dir, "metrics_over_time.png")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()
        print(f"График метрик во времени сохранен: {fig_path}")

    def create_comparative_dashboard(self):
        """
        Создает информационную панель с несколькими ключевыми графиками.
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Верхний левый график: Время ожидания
        comp_wait = self.analyzer.compare_waiting_times()
        axs[0,0].bar(comp_wait.keys(), comp_wait.values(), color=['#4c72b0','#55a868'])
        axs[0,0].set_title("Среднее время ожидания")
        axs[0,0].set_ylabel("Время (сек)")
        
        # Верхний правый график: Очереди
        comp_queue = self.analyzer.compare_queue_lengths()
        axs[0,1].bar(comp_queue.keys(), comp_queue.values(), color=['#c44e52','#8172b3'])
        axs[0,1].set_title("Средняя длина очереди")
        axs[0,1].set_ylabel("Длина очереди")
        
        # Нижний левый график: Пропускная способность
        comp_through = self.analyzer.compare_throughput()
        axs[1,0].bar(comp_through.keys(), comp_through.values(), color=['#ccb974','#64b5cd'])
        axs[1,0].set_title("Пропускная способность")
        axs[1,0].set_ylabel("Единиц/сек")
        
        # Нижний правый график: Изменение метрик (используем график из метода plot_metrics_over_time)
        df = self.analyzer.smart_df.copy()
        if "time" in df.columns:
            smart_wait_cols = [col for col in df.columns if col.endswith("_waiting")]
            df['avg_waiting'] = df[smart_wait_cols].mean(axis=1)
            axs[1,1].plot(df["time"], df['avg_waiting'], label="Умный", color="green")
            axs[1,1].set_title("Время ожидания (умный алгоритм)")
            axs[1,1].set_xlabel("Время (сек)")
            axs[1,1].set_ylabel("Время ожидания (сек)")
        
        plt.tight_layout()
        dashboard_path = os.path.join(self.output_dir, "comparative_dashboard.png")
        plt.savefig(dashboard_path)
        plt.close()
        print(f"Информационная панель сохранена: {dashboard_path}")

    def save_all_visualizations(self):
        """Вызывает все функции построения графиков и сохраняет визуализации в output_dir."""
        self.plot_waiting_time_comparison()
        self.plot_queue_length_comparison()
        self.plot_throughput_comparison()
        self.plot_waiting_time_by_direction()
        self.plot_queue_length_by_direction()
        self.plot_traffic_density_heatmap()
        self.plot_phase_duration_distribution()
        self.plot_metrics_over_time()
        self.create_comparative_dashboard()

###############################################################################
# Класс AnimatedTrafficSimulation
###############################################################################
class AnimatedTrafficSimulation:
    """
    Анимирует результаты симуляций для традиционного и умного алгоритмов.
    Создает анимации движения транспорта, сравнительную анимацию и анимацию изменения ключевых метрик.
    
    Параметры:
      - traditional_simulation_data: данные моделирования традиционного алгоритма (DataFrame).
      - smart_simulation_data: данные моделирования умного алгоритма (DataFrame).
      - output_dir: директория для сохранения анимаций.
    """
    def __init__(self, traditional_simulation_data, smart_simulation_data, output_dir):
        self.traditional_data = traditional_simulation_data
        self.smart_data = smart_simulation_data
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_intersection_animation(self):
        """
        Создает простую анимацию изменения средней длины очереди на перекрестке со временем.
        """
        df = self.traditional_data.copy()
        if "time" not in df.columns:
            print("Нет столбца 'time' для создания анимации.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(df["time"].min(), df["time"].max())
        ax.set_ylim(0, df.filter(like="_queue").max().max()*1.1)
        ax.set_title("Динамика очереди (традиционный алгоритм)")
        ax.set_xlabel("Время (сек)")
        ax.set_ylabel("Длина очереди")
        
        xdata, ydata = [], []
        queue_cols = [col for col in df.columns if col.endswith("_queue")]
        # Будем анимировать среднюю длину очереди
        avg_queue = df[queue_cols].mean(axis=1)

        def init():
            line.set_data([], [])
            return line,
        
        def update(frame):
            xdata.append(df["time"].iloc[frame])
            ydata.append(avg_queue.iloc[frame])
            line.set_data(xdata, ydata)
            return line,
        
        ani = FuncAnimation(fig, update, frames=range(len(df)), init_func=init, blit=True)
        ani_path = os.path.join(self.output_dir, "intersection_animation.mp4")
        writer = FFMpegWriter(fps=10)
        ani.save(ani_path, writer=writer)
        plt.close(fig)
        print(f"Анимация движения транспорта сохранена: {ani_path}")
    
    def create_comparative_animation(self):
        """
        Создает сравнительную анимацию среднего времени ожидания для традиционного и умного алгоритмов.
        """
        df_trad = self.traditional_data.copy()
        df_smart = self.smart_data.copy()
        if "time" not in df_trad.columns or "time" not in df_smart.columns:
            print("Нет столбца 'time' для создания сравнительной анимации.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        line1, = ax.plot([], [], label="Традиционный", color="blue", lw=2)
        line2, = ax.plot([], [], label="Умный", color="green", lw=2)
        ax.set_xlim(df_trad["time"].min(), df_trad["time"].max())
        # Определяем максимальное значение среднего времени ожидания
        trad_wait = df_trad.filter(like="_waiting").mean(axis=1)
        smart_wait = df_smart.filter(like="_waiting").mean(axis=1)
        y_max = max(trad_wait.max(), smart_wait.max())*1.1
        ax.set_ylim(0, y_max)
        ax.set_title("Сравнение среднего времени ожидания")
        ax.set_xlabel("Время (сек)")
        ax.set_ylabel("Время ожидания (сек)")
        ax.legend()
        
        xdata, ydata1, ydata2 = [], [], []
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2
        
        def update(frame):
            xdata.append(df_trad["time"].iloc[frame])
            ydata1.append(trad_wait.iloc[frame])
            ydata2.append(smart_wait.iloc[frame])
            line1.set_data(xdata, ydata1)
            line2.set_data(xdata, ydata2)
            return line1, line2
        
        ani = FuncAnimation(fig, update, frames=range(len(df_trad)), init_func=init, blit=True)
        ani_path = os.path.join(self.output_dir, "comparative_animation.mp4")
        writer = FFMpegWriter(fps=10)
        ani.save(ani_path, writer=writer)
        plt.close(fig)
        print(f"Сравнительная анимация сохранена: {ani_path}")

    def create_metrics_animation(self):
        """
        Создает анимацию изменения ключевых метрик (например, средней длины очереди) во времени.
        """
        df = self.smart_data.copy()
        if "time" not in df.columns:
            print("Нет столбца 'time' для создания анимации метрик.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(df["time"].min(), df["time"].max())
        queue_cols = [col for col in df.columns if col.endswith("_queue")]
        avg_queue = df[queue_cols].mean(axis=1)
        ax.set_ylim(0, avg_queue.max()*1.1)
        ax.set_title("Изменение средней длины очереди (умный алгоритм)")
        ax.set_xlabel("Время (сек)")
        ax.set_ylabel("Длина очереди")
        
        xdata, ydata = [], []
        
        def init():
            line.set_data([], [])
            return line,
        
        def update(frame):
            xdata.append(df["time"].iloc[frame])
            ydata.append(avg_queue.iloc[frame])
            line.set_data(xdata, ydata)
            return line,
        
        ani = FuncAnimation(fig, update, frames=range(len(df)), init_func=init, blit=True)
        ani_path = os.path.join(self.output_dir, "metrics_animation.mp4")
        writer = FFMpegWriter(fps=10)
        ani.save(ani_path, writer=writer)
        plt.close(fig)
        print(f"Анимация изменения метрик сохранена: {ani_path}")
    
    def save_animations(self):
        """Создает и сохраняет все анимации в output_dir."""
        self.create_intersection_animation()
        self.create_comparative_animation()
        self.create_metrics_animation()

###############################################################################
# Класс StatisticalAnalysis
###############################################################################
class StatisticalAnalysis:
    """
    Выполняет статистический анализ результатов моделирования.
    
    Параметры:
      - traditional_results_path: путь к CSV-файлу традиционной симуляции.
      - smart_results_path: путь к CSV-файлу умной симуляции.
      - output_dir: директория для сохранения отчетов.
    """
    def __init__(self, traditional_results_path, smart_results_path, output_dir):
        self.traditional_df = pd.read_csv(traditional_results_path)
        self.smart_df = pd.read_csv(smart_results_path)
        self.output_dir = output_dir
        
    def perform_t_test(self, metric_suffix="_waiting"):
        """
        Выполняет t-тест для сравнения средних значений метрики (например, по столбцам *_waiting)
        между традиционной и умной симуляциями.
        
        Возвращает словарь с p-значениями для каждого найденного столбца.
        """
        results = {}
        trad_cols = [col for col in self.traditional_df.columns if col.endswith(metric_suffix)]
        for col in trad_cols:
            if col in self.smart_df.columns:
                stat, pvalue = stats.ttest_ind(self.traditional_df[col].dropna(), 
                                               self.smart_df[col].dropna(), equal_var=False)
                results[col] = pvalue
        return results

    def calculate_improvement_percentage(self, metric_name):
        """
        Рассчитывает процентное улучшение метрики: ((trad - smart) / trad * 100).
        
        metric_name: ключ метрики ('avg_waiting_time', 'avg_queue_length', 'throughput', и т.д.)
        """
        # Для демонстрации мы берем средние значения из заранее рассчитанных показателей.
        # Здесь ожидается, что такие показатели доступны в виде средних (например, полученных из TrafficMetricsAnalyzer)
        # Если отсутствуют, этот метод можно доработать.
        # Для примера:
        traditional = {
            'avg_waiting_time': np.mean(self.traditional_df.filter(regex="_waiting").values.flatten()),
            'avg_queue_length': np.mean(self.traditional_df.filter(regex="_queue").values.flatten())
        }
        smart = {
            'avg_waiting_time': np.mean(self.smart_df.filter(regex="_waiting").values.flatten()),
            'avg_queue_length': np.mean(self.smart_df.filter(regex="_queue").values.flatten())
        }
        if traditional.get(metric_name, 0) == 0:
            return None
        improvement = (traditional[metric_name] - smart[metric_name]) / traditional[metric_name] * 100
        return improvement

    def calculate_confidence_intervals(self, values, confidence=0.95):
        """
        Рассчитывает доверительный интервал для массива значений.
        
        values: массив значений.
        Возвращает (mean, lower_bound, upper_bound).
        """
        a = 1.0 * np.array(values)
        n = len(a)
        mean_val = np.mean(a)
        se = stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return mean_val, mean_val - h, mean_val + h

    def generate_statistical_report(self):
        """
        Генерирует текстовый отчет со статистическим анализом, включая результаты t-теста,
        процент улучшения и доверительные интервалы.
        Сохраняет результат в файл report_statistical.txt.
        """
        t_test_results = self.perform_t_test(metric_suffix="_waiting")
        improvement_wait = self.calculate_improvement_percentage("avg_waiting_time")
        improvement_queue = self.calculate_improvement_percentage("avg_queue_length")
        
        report_lines = [
            "Статистический отчет",
            "----------------------------",
            "t-тест (время ожидания):",
        ]
        for col, pval in t_test_results.items():
            report_lines.append(f"  {col}: p-value = {pval:.4f}")
        report_lines.append("")
        report_lines.append(f"Процентное улучшение времени ожидания: {improvement_wait:.2f}%")
        report_lines.append(f"Процентное улучшение длины очереди: {improvement_queue:.2f}%")
        
        # Для демонстрации вычислим доверительный интервал для времени ожидания (объединенные данные)
        waiting_values = self.traditional_df.filter(regex="_waiting").values.flatten()
        mean_w, lower_w, upper_w = self.calculate_confidence_intervals(waiting_values)
        report_lines.append("")
        report_lines.append(f"Доверительный интервал для времени ожидания (традиционный): {lower_w:.2f} - {upper_w:.2f} (среднее: {mean_w:.2f})")
        
        report = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, "report_statistical.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Статистический отчет сохранен: {report_path}")
        return report

###############################################################################
# Функция main()
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Визуализация и сравнение результатов моделирования работы светофоров.")
    parser.add_argument("--traditional_results", type=str, required=True,
                        help="Путь к файлу с результатами традиционного алгоритма")
    parser.add_argument("--smart_results", type=str, required=True,
                        help="Путь к файлу с результатами умного алгоритма")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                        help="Директория для сохранения результатов (по умолчанию './comparison_results')")
    parser.add_argument("--create_animations", action="store_true",
                        help="Флаг для создания анимаций")
    parser.add_argument("--detailed_analysis", action="store_true",
                        help="Флаг для проведения детального статистического анализа")
    args = parser.parse_args()
    
    # Создаем выходную директорию, если её нет
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Создаем экземпляр TrafficMetricsAnalyzer, загружаем данные и рассчитываем метрики
    analyzer = TrafficMetricsAnalyzer(args.traditional_results, args.smart_results, args.output_dir)
    analyzer.load_data()
    analyzer.calculate_metrics()
    report = analyzer.generate_summary_report()
    print(report)
    
    # Создаем и сохраняем визуализации
    visualizer = TrafficVisualization(analyzer, args.output_dir)
    visualizer.save_all_visualizations()
    
    # Если флаг создания анимаций установлен – создаем анимации
    if args.create_animations:
        anim = AnimatedTrafficSimulation(analyzer.traditional_df, analyzer.smart_df, args.output_dir)
        anim.save_animations()
    
    # Если флаг детального анализа установлен — проводим статистический анализ
    if args.detailed_analysis:
        stats_analyzer = StatisticalAnalysis(args.traditional_results, args.smart_results, args.output_dir)
        stat_report = stats_analyzer.generate_statistical_report()
        print(stat_report)

if __name__ == "__main__":
    main()
