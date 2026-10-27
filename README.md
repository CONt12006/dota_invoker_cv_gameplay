<div align="center">


# Invoker CV 

### Играй за Invoker'а в Dota 2 жестами рук

Покажи жест в вебку — нейросеть найдёт руки, распознает позу<br>
и нажмёт Q / W / E / R / D / F за тебя. Никаких макросов: один жест — одна клавиша.

<br>

<img src="https://img.shields.io/badge/Dota_2-Invoker-B22222?style=for-the-badge&logo=steam&logoColor=white" alt="Dota 2 Invoker">

<br><br>

<img src="https://img.shields.io/badge/жесты-6_+_none-orange?style=flat-square" alt="gestures">
<img src="https://img.shields.io/badge/Python-3.10–3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/OpenCV-4.11-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/MediaPipe-Hands-0097A7?style=flat-square&logo=google&logoColor=white" alt="MediaPipe">
<img src="https://img.shields.io/badge/scikit--learn-MLP-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn">
<img src="https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Windows-pydirectinput-0078D6?style=flat-square&logo=windows&logoColor=white" alt="Windows">

</div>

---

## Жесты

| Жест      | Клавиша | Руки | Как показать                                                   | Автоповтор |
|-----------|:-------:|:----:|----------------------------------------------------------------|:----------:|
| `quas`    | `Q`     | 2    | Треугольник: указательные касаются сверху, большие — снизу      | 450 мс     |
| `wex`     | `W`     | 2    | Два кулака рядом, оба больших пальца вверх                      | 450 мс     |
| `exort`   | `E`     | 2    | Указательные горизонтально, кончики касаются, большие вверх     | 450 мс     |
| `invoke`  | `R`     | 2    | Ладони вместе, пальцы вверх («домик»)                           | —          |
| `spell_d` | `D`     | 1    | Одна рука, указательный палец вверх                             | —          |
| `spell_f` | `F`     | 1    | Одна рука, два пальца вверх (V)                                 | —          |
| `none`    | —       | —    | Руки опущены / случайные движения / переходы между жестами       | —          |

Пока держишь сферу (`quas`/`wex`/`exort`), клавиша повторяется каждые `repeat_ms`. Чтобы нажать тот же жест ещё раз без повтора — опусти руки или смени жест.

## Как это работает

1. **`hands.py`** — MediaPipe Hands находит до двух рук, из кадра собирается вектор на 156 признаков:
   - форма левой и правой кисти (2 × 63) — точки относительно запястья, нормированные по размеру ладони, поэтому не зависят от положения руки в кадре и расстояния до камеры;
   - положение правого запястья относительно левого (3);
   - попарные расстояния между кончиками пальцев двух рук (25) — чтобы ловить «пальцы касаются»;
   - флаги «рука найдена» (2).
2. **`train.py`** — `StandardScaler` + `MLPClassifier(128, 64)` из scikit-learn. Обучение за секунды на CPU. Аугментации: шум, масштаб, поворот кисти ±10°, перестановка и зеркалирование одноруких кадров (MediaPipe иногда путает левую и правую руку).
3. **`controller.py`** — жест засчитывается, если держится `stable_frames` кадров подряд с уверенностью ≥ `threshold`. Двуручные жесты не срабатывают, если в кадре одна рука. Нажатия идут через `pydirectinput` (SendInput со скан-кодами — игра их видит).

## Установка

Нужен Python 3.10–3.12 (под него собраны `mediapipe 0.10.21` и `numpy 1.26`) и вебка.

```bash
git clone https://github.com/CONt12006/dota_invoker_cv_gameplay.git
cd dota_invoker_cv_gameplay
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

> Нажатия в игру отправляются через `pydirectinput`, он работает только на **Windows**. На Linux контроллер можно запускать с `--dry-run` (распознавание без нажатий).

## Быстрый старт

В репозитории уже есть записанные жесты (`data/dota_gestures/`) и обученная модель (`models/dota_gestures.joblib`), так что можно сразу запускать:

```bash
python controller.py --dry-run   # проверить распознавание, ничего не нажимает
python controller.py             # боевой режим, стартует на ПАУЗЕ
```

- **F8** (глобально) или **P** в окне — включить / поставить на паузу.
- **Q** / **Esc** в окне — выход.
- `--armed` — стартовать сразу включённым.
- `--camera N` — выбрать другую камеру.

В Dota поставь режим **«Оконный без рамки» (borderless)** — тогда окно контроллера будет висеть поверх игры.

## Свои жесты

Модель обучена на моих руках и моей камере — у тебя лучше всего будет работать, если записать свои.

### 1. Запись

```bash
python record.py              # по умолчанию сессия 6 секунд
python record.py --seconds 10
```

| Клавиша     | Действие                                      |
|-------------|-----------------------------------------------|
| `0`–`9`     | выбрать жест (`0` — `none`)                   |
| `n` / `p`   | следующий / предыдущий жест                   |
| `Space`     | записать сессию (1 с подготовки + запись)     |
| `u`         | удалить последнюю сессию                      |
| `q` / `Esc` | выход                                         |

Советы:

- во время записи держи жест и слегка меняй его: поворачивай кисть, двигай ближе/дальше, выше/ниже;
- для `none` опускай руки, чеши нос, показывай переходы между жестами — это учит модель **не** нажимать клавиши случайно;
- желательно 300+ кадров и 3+ сессии на каждый жест.

Каждая сессия сохраняется в `data/dota_gestures/<жест>/<timestamp>.npy` (массив `кадры × 156`).

### 2. Обучение

```bash
python train.py
```

Скрипт печатает отчёт по качеству на **незнакомых сессиях** (`GroupKFold` по записям — соседние кадры одной сессии почти одинаковые и завысили бы точность), матрицу ошибок и сохраняет модель в `models/dota_gestures.joblib`.

### 3. Запуск

```bash
python controller.py
```

## Настройка — `config.json`

```json
{
  "camera": 0,
  "threshold": 0.80,
  "stable_frames": 4,
  "toggle_hotkey": "<f8>",
  "gestures": {
    "quas": {"key": "q", "hands": 2, "repeat_ms": 450, "hint": "..."}
  }
}
```

| Поле            | Что делает                                                                  |
|-----------------|-----------------------------------------------------------------------------|
| `camera`        | индекс вебки                                                                |
| `threshold`     | минимальная уверенность модели, чтобы кадр засчитался                       |
| `stable_frames` | сколько кадров подряд жест должен держаться (больше — надёжнее, но медленнее) |
| `toggle_hotkey` | глобальная горячая клавиша паузы (формат `pynput`)                          |
| `key`           | какую клавишу жать                                                          |
| `hands`         | сколько рук нужно жесту                                                     |
| `repeat_ms`     | интервал автоповтора при удержании, `0` — без повтора                       |
| `hint`          | подсказка в окне записи                                                     |

Чтобы добавить новый жест — допиши его в `gestures`, запиши в `record.py` и переобучи `train.py`.

## Структура

```
.
├── hands.py          # детекция рук, признаки, аугментации, отрисовка
├── record.py         # запись датасета с вебки
├── train.py          # обучение и валидация MLP
├── controller.py     # жест -> нажатие клавиши
├── config.json       # жесты, клавиши, пороги
├── data/dota_gestures/   # записанные сессии (.npy)
└── models/dota_gestures.joblib
```

## Стек

Python · OpenCV · MediaPipe Hands · NumPy · scikit-learn · pydirectinput · pynput
