#!/usr/bin/env python3
"""
traditional_traffic_light.py

Моделирование работы традиционных светофоров с фиксированными фазами.
Используются OpenCV для отрисовки и подсчёта.

Зависимости:
  pip install opencv-python numpy

Пример запуска:
  python traditional_traffic_light.py
"""

import cv2
import numpy as np
import argparse

class TrafficLightPhase:
    def __init__(self, phase_id, green_directions, duration, yellow_time=3):
        self.phase_id = phase_id
        self.green_directions = green_directions
        self.duration = duration
        self.yellow_time = yellow_time

    def get_state(self, elapsed_time):
        state = {}
        for d in ["N", "S", "E", "W"]:
            if d in self.green_directions:
                state[d] = "yellow" if elapsed_time >= self.duration - self.yellow_time else "green"
            else:
                state[d] = "red"
        return state

class TraditionalTrafficLightController:
    def __init__(self, phases, cycle_time):
        self.phases = phases
        self.cycle_time = cycle_time
        self.current_time = 0.0

    def update(self, dt):
        self.current_time = (self.current_time + dt) % self.cycle_time

    def get_current_phase(self):
        elapsed = 0.0
        for phase in self.phases:
            if self.current_time < elapsed + phase.duration:
                local_time = self.current_time - elapsed
                return phase, local_time
            elapsed += phase.duration
        return self.phases[-1], self.current_time - (elapsed - self.phases[-1].duration)

    def get_current_state(self):
        phase, elapsed = self.get_current_phase()
        return phase.get_state(elapsed)

class IntersectionWithTraditionalLights:
    def __init__(self, lanes, controller):
        self.lanes = lanes  # словарь с ключами, например, "N", "S", "E", "W"
        self.controller = controller

    def update(self, dt):
        self.controller.update(dt)
        return self.controller.get_current_state()

def main():
    parser = argparse.ArgumentParser(description="Традиционный контроллер светофоров")
    args = parser.parse_args()

    phase1 = TrafficLightPhase("Phase1", ["N", "S"], 30, yellow_time=3)
    phase2 = TrafficLightPhase("Phase2", ["E", "W"], 30, yellow_time=3)
    controller = TraditionalTrafficLightController([phase1, phase2], cycle_time=60)

    # Простейшая визуализация: создадим окно, где будем отображать состояние светофоров.
    while True:
        controller.update(0.1)
        state = controller.get_current_state()
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        y = 50
        for direction, color in state.items():
            cv2.putText(img, f"{direction}: {color}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if color=="green" else (0,0,255), 2)
            y += 50
        cv2.imshow("Traditional Traffic Lights", img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
