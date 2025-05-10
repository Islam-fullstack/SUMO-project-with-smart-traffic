#!/usr/bin/env python3
"""
visualization_and_comparison.py

Визуализация и сравнительный анализ результатов моделирования работы светофоров.
Строятся графики сравнения среднего времени ожидания, длины очередей и пропускной способности,
с использованием Matplotlib, Seaborn и Pandas.

Зависимости:
  pip install numpy pandas matplotlib seaborn scipy

Пример запуска:
  python visualization_and_comparison.py --traditional_results results/simulation_results.csv --smart_results results/simulation_results.csv --output_dir ./results
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class TrafficMetricsAnalyzer:
    def __init__(self, traditional_results_path, smart_results_path, output_dir):
        self.traditional_results_path = traditional_results_path
        self.smart_results_path = smart_results_path
        self.output_dir = output_dir
        self.traditional_df = None
        self.smart_df = None
        self.traditional_metrics = {}
        self.smart_metrics = {}

    def load_data(self):
        self.traditional_df = pd.read_csv(self.traditional_results_path)
        self.smart_df = pd.read_csv(self.smart_results_path)
        print("[INFO] Данные загружены.")

    def calculate_metrics(self):
        def compute_avg_metric(df, metric_suffix):
            cols = [col for col in df.columns if col.endswith(metric_suffix)]
            if not cols:
                return None
            return df[cols].mean().mean()
        def compute_throughput(df):
            throughput_vals = {}
            queue_cols = [col for col in df.columns if col.endswith("_queue")]
            for col in queue_cols:
                diffs = df[col].diff().fillna(0)
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
        print("[INFO] Метрики рассчитаны.")

    def compare_waiting_times(self):
        return {'traditional': self.traditional_metrics.get('avg_waiting_time', np.nan),
                'smart': self.smart_metrics.get('avg_waiting_time', np.nan)}

    def compare_queue_lengths(self):
        return {'traditional': self.traditional_metrics.get('avg_queue_length', np.nan),
                'smart': self.smart_metrics.get('avg_queue_length', np.nan)}

    def compare_throughput(self):
        return {'traditional': self.traditional_metrics.get('throughput', np.nan),
                'smart': self.smart_metrics.get('throughput', np.nan)}

    def generate_summary_report(self):
        report_lines = [
            "Сводный отчет по моделированию работы светофоров",
            "--------------------------------------------------------",
            f"Среднее время ожидания (сек): традиционный = {self.traditional_metrics.get('avg_waiting_time', 'N/A'):.2f}, умный = {self.smart_metrics.get('avg_waiting_time', 'N/A'):.2f}",
            f"Средняя длина очереди: традиционный = {self.traditional_metrics.get('avg_queue_length', 'N/A'):.2f}, умный = {self.smart_metrics.get('avg_queue_length', 'N/A'):.2f}",
            f"Пропускная способность (ед/сек): традиционный = {self.traditional_metrics.get('throughput', 'N/A'):.2f}, умный = {self.smart_metrics.get('throughput', 'N/A'):.2f}"
        ]
        report = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, "summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[INFO] Отчет сохранен в {report_path}")
        return report

class TrafficVisualization:
    def __init__(self, analyzer, output_dir):
        self.analyzer = analyzer
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_waiting_time_comparison(self):
        comp = self.analyzer.compare_waiting_times()
        plt.figure(figsize=(6,4))
        sns.barplot(x=list(comp.keys()), y=list(comp.values()), palette="viridis")
        plt.title("Сравнение среднего времени ожидания")
        plt.ylabel("Время ожидания (сек)")
        fig_path = os.path.join(self.output_dir, "waiting_time_comparison.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"[INFO] График времени ожидания сохранен: {fig_path}")

    def plot_queue_length_comparison(self):
        comp = self.analyzer.compare_queue_lengths()
        plt.figure(figsize=(6,4))
        sns.barplot(x=list(comp.keys()), y=list(comp.values()), palette="magma")
        plt.title("Сравнение средней длины очередей")
        plt.ylabel("Длина очереди")
        fig_path = os.path.join(self.output_dir, "queue_length_comparison.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"[INFO] График длины очередей сохранен: {fig_path}")

    def plot_throughput_comparison(self):
        comp = self.analyzer.compare_throughput()
        plt.figure(figsize=(6,4))
        sns.barplot(x=list(comp.keys()), y=list(comp.values()), palette="cubehelix")
        plt.title("Сравнение пропускной способности")
        plt.ylabel("Пропускная способность (ед/сек)")
        fig_path = os.path.join(self.output_dir, "throughput_comparison.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"[INFO] График пропускной способности сохранен: {fig_path}")

    def save_all_visualizations(self):
        self.plot_waiting_time_comparison()
        self.plot_queue_length_comparison()
        self.plot_throughput_comparison()

def main():
    parser = argparse.ArgumentParser(description="Визуализация и анализ результатов моделирования")
    parser.add_argument("--traditional_results", type=str, required=True,
                        help="Путь к файлу результатов традиционного алгоритма")
    parser.add_argument("--smart_results", type=str, required=True,
                        help="Путь к файлу результатов умного алгоритма")
    parser.add_argument("--output_dir", type=str, default="./results", help="Выходная директория")
    args = parser.parse_args()

    analyzer = TrafficMetricsAnalyzer(args.traditional_results, args.smart_results, args.output_dir)
    analyzer.load_data()
    analyzer.calculate_metrics()
    report = analyzer.generate_summary_report()
    print(report)
    visualizer = TrafficVisualization(analyzer, args.output_dir)
    visualizer.save_all_visualizations()

if __name__ == "__main__":
    main()
