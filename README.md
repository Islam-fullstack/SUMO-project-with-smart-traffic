```markdown
# Система умных светофоров: моделирование и анализ
```

## 1. Название проекта и краткое описание

```
**Название проекта:**
**Система умных светофоров: моделирование и анализ**
```

**Краткое описание:**  
Данный проект направлен на создание системы умных светофоров, способной моделировать дорожный поток, оптимизировать управление светофорами, а также проводить сравнительный анализ традиционных и адаптивных алгоритмов управления. Система использует данные обнаружения транспорта (с видео, симуляции или интеграции с SUMO), моделирует трафик, и динамически регулирует фазы светофоров для снижения заторов и оптимизации пропускной способности.

**Концепция умных светофоров:**  
Умные светофоры анализируют плотность транспортного потока, длину очередей и время ожидания на перекрестках. На основе этих данных система адаптирует продолжительности зеленых фаз, уменьшая заторы и повышая эффективность движения.

---

## 2. Обзор системы

### Архитектура и основные компоненты:

- **Модуль обнаружения транспорта:**  
  Использует алгоритмы глубокого обучения (например, YOLOv8) для обнаружения транспортных средств из видеопотока, симуляционных данных или данных SUMO.
- **Модуль моделирования трафика:**  
  Симулирует движение транспортных средств, учитывая реальные параметры (скорость, ускорение, длину транспортного средства).
- **Контроллеры светофоров:**
  - _Традиционный контроллер_ с фиксированными фазами
  - _Умный контроллер_ с адаптивными фазами, подстраивающимися под текущий трафик
- **Модуль интеграции с SUMO:**  
  Позволяет внедрить систему в микроскопическую симуляцию дорожного движения SUMO с использованием TraCI и sumolib.
- **Модуль визуализации и сравнительного анализа:**  
  Генерирует графики, диаграммы и анимации для оценки эффективности алгоритмов.

### Схема взаимодействия компонентов:

```

         +---------------------+
         | Обнаружение         |
         | транспорта          |<------------------+
         +----------+----------+                   |
                    |                              |
                    v                              |
         +----------+----------+         +---------+-----------+
         | Модуль моделирования|         |  Интеграция с SUMO  |
         | трафика             |         |  (TraCI, sumolib)   |
         +----------+----------+         +---------+-----------+
                    |                              ^
                    v                              |
         +----------+----------+                   |
         | Контроллеры         |-------------------+
         | светофоров          |
         | (традиционный и умный)|
         +----------+----------+
                    |
                    v
         +----------+----------+
         | Визуализация и      |
         | сравнительный анализ|
         +---------------------+

```

```
### Поддерживаемые режимы работы:
```

- `simulation` – моделирование трафика в симуляторе
- `sumo_integration` – интеграция с SUMO для микроскопической симуляции
- `real_time` – обработка видео в реальном времени

---

```
## 3. Требования к системе
```

### Аппаратное обеспечение:

- **Процессор:** двухъядерный Intel/AMD (минимум 2.0 ГГц)
- **Оперативная память:** минимум 4 ГБ (рекомендуется 8 ГБ)
- **Хранилище:** минимум 1 ГБ свободного места

### Поддерживаемые операционные системы:

- Windows 10+
- Linux (например, Ubuntu 18.04+)
- macOS

### Необходимые зависимости (версии рекомендуется обновлять):

- Python 3.8+
- numpy
- pandas
- matplotlib
- opencv-python
- ultralytics (YOLOv8)
- sumolib, traci
- seaborn
- pyyaml

---

```
## 4. Установка
```

### Установка зависимостей:

```bash
pip install numpy pandas matplotlib opencv-python ultralytics sumolib traci seaborn pyyaml
```

### Установка SUMO (если требуется):

1. Скачайте SUMO с [официального сайта SUMO](https://www.eclipse.org/sumo/).
2. Установите SUMO согласно документации.
3. Добавьте путь к SUMO в переменную среды `PATH`.

### Клонирование репозитория:

```bash
git clone https://github.com/yourusername/smart-traffic-light-system.git
cd smart-traffic-light-system
```

### Настройка виртуального окружения:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 5. Быстрый старт

### Запуск симуляции:

```bash
python main.py --config config.json --mode simulation --detection_source simulation --output_dir ./results
```

### Запуск интеграции с SUMO:

```bash
python main.py --config config.json --mode sumo_integration --detection_source sumo --output_dir ./results
```

### Запуск обработки видео в реальном времени:

```bash
python main.py --config config.json --mode real_time --detection_source video --output_dir ./results
```

---

## 6. Настройка

### Конфигурационный файл (`config.json`):

Конфигурационный файл содержит несколько основных разделов:

- **general_settings:** настройки системы (режим работы, источник данных, output_dir, уровень логирования, случайное семя).
- **simulation_settings:** параметры симуляции (время, dt, количество перекрестков, интенсивности, распределение типов, длины полос, максимальные скорости).
- **detection_settings:** настройки обнаружения транспорта (источник видео, модель YOLO, порог уверенности, классы, ROI, FPS, флаг сохранения видео).
- **traditional_controller_settings:** настройки традиционного контроллера (фазы, желтый сигнал, общий цикл).
- **smart_controller_settings:** настройки умного контроллера (минимальные/максимальные длительности фаз, лимиты цикла, параметры оптимизации).
- **sumo_integration_settings:** настройки интеграции с SUMO (файл конфигурации, порт TraCI, светофоры, диапазон детектирования, OSM-файл).
- **visualization_settings:** настройки визуализации (типы графиков, форматы, анимации, метрики, стиль, dpi).

Детальный пример `config.json` можно найти в корне репозитория.

### Примеры настройки:

- Для симуляции: `"mode": "simulation"`, `"detection_source": "simulation"`.
- Для интеграции с SUMO: `"mode": "sumo_integration"` и заполнение параметров в `sumo_integration_settings`.
- Для обработки видео: `"mode": "real_time"` и `"detection_source": "video"`.

---

## 7. Структура проекта

```
smart-traffic-light-system/
├── config.json                  # Конфигурационный файл системы
├── main.py                      # Главный модуль системы
├── vehicle_detection.py         # Обнаружение транспорта (YOLOv8)
├── traffic_simulation.py        # Моделирование трафика
├── traditional_traffic_light.py # Традиционный контроллер светофоров
├── smart_traffic_light.py       # Умный контроллер светофоров
├── sumo_integration.py          # Интеграция с SUMO
├── visualization_and_comparison.py  # Визуализация и сравнение результатов
├── requirements.txt             # Список зависимостей
├── README.md                    # Документация проекта
└── docs/                        # Дополнительная документация (например, UML-диаграммы)
```

### Основные классы:

- **SmartTrafficLightSystem:** объединяет и управляет всеми компонентами.
- **ConfigManager:** загружает и валидирует конфигурацию.
- **SystemManager:** инициализирует, запускает и контролирует выполнение системы.
- **DataExporter:** экспортирует результаты системы в различные форматы.
- Дополнительные классы для обнаружения, симуляции, управления светофорами и интеграции с SUMO.

---

## 8. Использование API

### Примеры использования:

```python
from vehicle_detection import VehicleDetector

# Инициализация детектора YOLOv8
detector = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
detector.load_model()
frame = ...  # Получите кадр из видео или симуляции
processed_frame, detections = detector.process_frame(frame)
```

### Создание собственных алгоритмов:

Расширьте базовые классы `SmartTrafficLightController` или `TraditionalTrafficLightController` для реализации новых алгоритмов управления светофорами.

### Интеграция:

Используйте API системы для интеграции с внешними системами или сенсорными данными.

---

## 9. Визуализация результатов

### Доступные типы графиков:

- **Bar charts:** сравнение среднего времени ожидания и длины очередей.
- **Line graphs:** динамика метрик во времени.
- **Heatmaps:** распределение плотности транспорта.
- **KDE-графики:** распределение значений по направлениям.
- **Анимации:** динамическое отображение изменения ключевых показателей.

### Интерпретация результатов:

Сравнительный анализ графиков позволяет оценить преимущества умного управления по сравнению с традиционным (например, снижение времени ожидания, улучшенная пропускная способность).

---

## 10. Интеграция с SUMO

### Инструкции по настройке SUMO:

1. Установите SUMO согласно официальной документации.
2. Укажите путь к конфигурационному файлу SUMO в разделе `sumo_integration_settings` в `config.json`.
3. Обеспечьте корректную настройку TraCI (порт, использование GUI).

### Создание сценариев:

Используйте модуль `sumo_integration.py` или `SUMOScenarioGenerator` для генерации сети и спроса на транспорт на основе OSM.

---

## 11. Расширение функциональности

- **Новые алгоритмы:**  
  Расширьте классы контроллеров светофоров для реализации новых алгоритмов управления.
- **Дополнительные визуализации:**  
  Добавьте новые типы графиков с использованием Matplotlib и Seaborn.
- **Новые источники данных:**  
  Интегрируйте дополнительные сенсоры или API для получения данных о трафике.

---

## 12. Часто задаваемые вопросы (FAQ)

- **Как изменить режим работы системы?**  
  Отредактируйте параметр `mode` в конфигурационном файле (`config.json`) на `"simulation"`, `"sumo_integration"` или `"real_time"`.
- **Как настроить параметры обнаружения транспорта?**  
  Измените раздел `detection_settings` в `config.json`, установив верный путь к модели YOLO, порог уверенности и координаты ROI.
- **Почему не срабатывает детекция транспортных средств?**  
  Проверьте правильность заданных параметров модели, порога и ROI.
- **Как запустить интеграцию с SUMO?**  
  Убедитесь, что SUMO установлен, а параметры в разделе `sumo_integration_settings` заполнены корректно.

---

## 13. Лицензия

Этот проект распространяется под лицензией **MIT**. Подробнее см. в файле [LICENSE](LICENSE).

---

## 14. Авторы и контакты

**Основной автор:**

- Islam

**Контактная информация:**

- Email: example@example.com
- GitHub: [yourusername](https://github.com/yourusername)

**Благодарности:**

- Сообщество SUMO
- Разработчики ultralytics (YOLOv8)
- OpenStreetMap и все, кто внёс вклад в развитие транспортных технологий

---

### Пример использования конфигурационного файла:

```python
import json

# Загрузка конфигурации
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Доступ к общим настройкам
mode = config["general_settings"]["mode"]
output_dir = config["general_settings"]["output_dir"]

print(f"Режим работы системы: {mode}")
print(f"Результаты будут сохранены в: {output_dir}")
```

### Инструкции по модификации:

- Откройте файл `config.json` в текстовом редакторе.
- Измените значения параметров в соответствующих разделах, чтобы настроить систему под свои нужды.
- Сохраните изменения и запустите систему с помощью `python main.py --config config.json ...`

---

This README provides a detailed, comprehensive guide to setting up and using the Smart Traffic Light System. Follow the instructions to configure, run, and extend the project according to your specific requirements.

```

---

```
