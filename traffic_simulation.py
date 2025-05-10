#!/usr/bin/env python3
"""
traffic_simulation.py

Моделирование транспортного потока на участке дороги с перекрестками.
Используются NumPy для вычислений и Matplotlib для визуализации.

Зависимости:
  pip install numpy matplotlib pandas

Пример запуска:
  python traffic_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

class Vehicle:
    def __init__(self, vehicle_id, vehicle_type, direction, speed, position, acceleration, max_speed, length):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.direction = direction
        self.speed = speed
        self.position = position
        self.acceleration = acceleration
        self.max_speed = max_speed
        self.length = length
        self.waiting_time = 0.0

    def update(self, dt, traffic_light_state, leader_vehicle=None, lane_length=None):
        desired_speed = self.max_speed
        if traffic_light_state == "red" and lane_length is not None:
            distance_to_intersection = lane_length - (self.position + self.length)
            if distance_to_intersection < 20:
                desired_speed = 0
        if leader_vehicle:
            gap = leader_vehicle.position - (leader_vehicle.length + self.position)
            safe_gap = 5
            if gap < safe_gap:
                desired_speed = min(desired_speed, leader_vehicle.speed)
        if self.speed < desired_speed:
            self.speed = min(self.speed + self.acceleration * dt, desired_speed, self.max_speed)
        else:
            self.speed = max(self.speed - self.acceleration * dt, desired_speed, 0)
        if self.speed < 0.1:
            self.waiting_time += dt
        self.position += self.speed * dt
        return {
            'vehicle_id': self.vehicle_id,
            'vehicle_type': self.vehicle_type,
            'direction': self.direction,
            'position': self.position,
            'speed': self.speed,
            'waiting_time': self.waiting_time
        }

class TrafficLane:
    def __init__(self, lane_id, direction, length):
        self.lane_id = lane_id
        self.direction = direction
        self.length = length
        self.vehicles = []

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def update_vehicles(self, dt, traffic_light_state):
        self.vehicles.sort(key=lambda v: v.position)
        updated_states = []
        for i, vehicle in enumerate(self.vehicles):
            leader = self.vehicles[i+1] if i < len(self.vehicles)-1 else None
            state = vehicle.update(dt, traffic_light_state, leader_vehicle=leader, lane_length=self.length)
            updated_states.append(state)
        self.vehicles = [v for v in self.vehicles if v.position <= self.length]
        return updated_states

    def get_density(self):
        return len(self.vehicles) / self.length

    def get_queue_length(self):
        threshold = 0.1
        if not self.vehicles:
            return 0
        sorted_vehicles = sorted(self.vehicles, key=lambda v: v.position)
        front_vehicle = sorted_vehicles[-1]
        if front_vehicle.speed < threshold:
            return max(self.length - (front_vehicle.position + front_vehicle.length), 0)
        return 0

class Intersection:
    def __init__(self, intersection_id, lanes):
        self.intersection_id = intersection_id
        self.lanes = lanes
        self.traffic_light_state = {}

    def set_traffic_light_states(self, states):
        self.traffic_light_state.update(states)

    def update(self, dt):
        for direction, lane in self.lanes.items():
            state = self.traffic_light_state.get(direction, "red")
            lane.update_vehicles(dt, state)
        return self.get_statistics()

    def get_statistics(self):
        stats = {}
        for direction, lane in self.lanes.items():
            stats[direction] = {"density": lane.get_density(), "queue_length": lane.get_queue_length()}
        return stats

class TrafficGenerator:
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

    def generate_vehicles(self, dt):
        new_vehicles = {direction: [] for direction in self.arrival_rates.keys()}
        for direction, rate in self.arrival_rates.items():
            num = np.random.poisson(rate * dt)
            for _ in range(num):
                vtype = np.random.choice(list(self.vehicle_types.keys()), p=list(self.vehicle_types.values()))
                params = self.vehicle_params[vtype]
                vehicle = Vehicle(self.vehicle_id_counter, vtype, direction, 0.0, 0.0,
                                  params["acceleration"], params["max_speed"], params["length"])
                self.vehicle_id_counter += 1
                new_vehicles[direction].append(vehicle)
        return new_vehicles

class TrafficSimulation:
    def __init__(self, intersections, traffic_generator, simulation_time, dt):
        self.intersections = intersections
        self.traffic_generator = traffic_generator
        self.simulation_time = simulation_time
        self.dt = dt
        self.statistics = []
        self.time_steps = []

    def run(self):
        current_time = 0.0
        light_cycle = 30
        while current_time < self.simulation_time:
            # Чередование фаз: на первой фазе зеленые направлены N и S, затем E и W
            if int(current_time // light_cycle) % 2 == 0:
                state = {"N": "green", "S": "green", "E": "red", "W": "red"}
            else:
                state = {"N": "red", "S": "red", "E": "green", "W": "green"}
            for intersection in self.intersections:
                intersection.set_traffic_light_states(state)
                new_vehicles = self.traffic_generator.generate_vehicles(self.dt)
                for direction, vehicles in new_vehicles.items():
                    lane = intersection.lanes.get(direction)
                    if lane:
                        for v in vehicles:
                            lane.add_vehicle(v)
                stats = intersection.update(self.dt)
                stats["time"] = current_time
                self.statistics.append(stats)
            self.time_steps.append(current_time)
            current_time += self.dt
        print("Симуляция завершена.")
        return self.statistics

    def visualize(self, output_path=None):
        times = [stat["time"] for stat in self.statistics]
        avg_density = []
        avg_queue = []
        for stat in self.statistics:
            dens = np.mean([stat[d]["density"] for d in ["N", "S", "E", "W"]])
            queue = np.mean([stat[d]["queue_length"] for d in ["N", "S", "E", "W"]])
            avg_density.append(dens)
            avg_queue.append(queue)
        plt.figure(figsize=(10,5))
        plt.plot(times, avg_density, label="Average Density", color="blue")
        plt.plot(times, avg_queue, label="Average Queue Length", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title("Traffic Metrics Over Time")
        plt.legend()
        if output_path:
            plt.savefig(output_path)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Моделирование транспортного потока")
    parser.add_argument("--sim_time", type=float, default=3600)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    lane_length = 500.0
    lanes = {
        "N": TrafficLane("N1", "N", lane_length),
        "S": TrafficLane("S1", "S", lane_length),
        "E": TrafficLane("E1", "E", lane_length),
        "W": TrafficLane("W1", "W", lane_length)
    }
    intersection = Intersection("I1", lanes)
    arrival_rates = {"N": 0.2, "S": 0.15, "E": 0.2, "W": 0.15}
    vehicle_types = {"car": 0.7, "truck": 0.15, "bus": 0.1, "motorcycle": 0.05}
    generator = TrafficGenerator(arrival_rates, vehicle_types)
    sim = TrafficSimulation([intersection], generator, args.sim_time, args.dt)
    sim.run()
    sim.visualize(output_path="simulation_results.png")

if __name__ == "__main__":
    main()
