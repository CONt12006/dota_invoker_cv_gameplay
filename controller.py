"""Управление Dota 2 жестами: жест с вебки -> нажатие клавиши.

python src/dota/controller.py            # старт на паузе, F8 — включить/выключить
python src/dota/controller.py --dry-run  # ничего не нажимает, только показывает

Как работает:
  1. Камера -> MediaPipe Hands -> 156 признаков кадра (hands.py).
  2. Модель (train.py) даёт вероятность каждого жеста.
  3. Жест считается показанным, если он стабильно держится stable_frames кадров подряд
     с уверенностью >= threshold. Тогда ОДИН раз нажимается его клавиша.
  4. Пока жест держится и у него repeat_ms > 0 — клавиша повторяется с этим интервалом
     (удобно для сфер Invoker: держишь «quas» — Q, Q, Q).
  5. Чтобы снова нажать тот же жест без повтора — опусти руки (none) или смени жест.

Одно срабатывание = одна клавиша: никаких автоматических комбо (это уже скрипт, за них банят).
В Dota включи «Оконный без рамки» (borderless), чтобы окно контроллера было видно поверх игры.
"""
import argparse
import time
from collections import deque

import cv2
import joblib
import numpy as np

from hands import (FEATURE_VERSION, MODEL_PATH, NONE, create_hands, detect, draw, features,
                   load_config, open_camera, read_frame, text)


class Trigger:
    """Превращает поток «жест + уверенность по кадрам» в нажатия клавиш."""

    def __init__(self, config):
        self.gestures = config["gestures"]
        self.threshold = float(config.get("threshold", 0.8))
        self.history = deque(maxlen=int(config.get("stable_frames", 4)))
        self.current = NONE
        self.last_fire = 0.0

    def update(self, label, prob, now):
        self.history.append(label if prob >= self.threshold else NONE)
        if len(self.history) < self.history.maxlen or len(set(self.history)) != 1:
            return None
        stable = self.history[0]
        if stable != self.current:                       # начало нового жеста
            self.current = stable
            if stable != NONE and stable in self.gestures:
                self.last_fire = now
                return self.gestures[stable]["key"]
            return None
        repeat = self.gestures.get(stable, {}).get("repeat_ms", 0)
        if stable != NONE and repeat and (now - self.last_fire) * 1000 >= repeat:
            self.last_fire = now
            return self.gestures[stable]["key"]
        return None


class Keyboard:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.pdi = None
        if not dry_run:
            try:
                import pydirectinput                     # SendInput со скан-кодами — игры его видят
                pydirectinput.PAUSE = 0                  # по умолчанию 0.1 с задержки после каждого нажатия
                pydirectinput.FAILSAFE = False
                self.pdi = pydirectinput
            except ImportError:
                print("pydirectinput не установлен (pip install pydirectinput) — работаю в режиме --dry-run")
                self.dry_run = True

    def press(self, key):
        if self.pdi:
            self.pdi.press(key)


def start_hotkey(hotkey, callback):
    try:
        from pynput import keyboard
    except ImportError:
        print("pynput не установлен — глобальной горячей клавиши нет, пауза только клавишей P в окне")
        return None
    listener = keyboard.GlobalHotKeys({hotkey: callback})
    listener.daemon = True
    listener.start()
    return listener


def main():
    config = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=config.get("camera", 0))
    ap.add_argument("--dry-run", action="store_true", help="не нажимать клавиши, только показывать")
    ap.add_argument("--armed", action="store_true", help="стартовать сразу включённым (по умолчанию пауза)")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Нет модели {MODEL_PATH}. Сначала: record.py, затем train.py")
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise SystemExit("Модель обучена на старых признаках — переобучи: python src/dota/train.py")
    model, labels = bundle["model"], bundle["labels"]
    unknown = [g for g in config["gestures"] if g not in labels]
    if unknown:
        print("Жесты из config.json, которых модель не знает (не будут срабатывать):", unknown)

    # сколько рук нужно каждому жесту: Q/W/E/R — две, D/F — одна
    need_hands = {g: int(v.get("hands", 1)) for g, v in config["gestures"].items()}
    trigger, kb = Trigger(config), Keyboard(args.dry_run)
    state = {"armed": args.armed}

    def toggle():
        state["armed"] = not state["armed"]
        trigger.history.clear()
        print("ВКЛЮЧЕНО" if state["armed"] else "ПАУЗА")

    hotkey = config.get("toggle_hotkey", "<f8>")
    start_hotkey(hotkey, toggle)

    win = "Dota gestures"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 480, 360)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)   # поверх игры (Windows)
    except Exception:
        pass

    cap = open_camera(args.camera)
    fps, prev, log = 0.0, time.time(), deque(maxlen=6)
    print(f"Старт: {'ВКЛЮЧЕНО' if state['armed'] else 'ПАУЗА'} — {hotkey} или P в окне переключает.")
    with create_hands() as hands:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            now = time.time()
            left, right, res = detect(hands, frame)
            n_hands = (left is not None) + (right is not None)
            if n_hands == 0:
                label, prob = NONE, 1.0
            else:
                p = model.predict_proba(features(left, right)[None])[0]
                i = int(p.argmax())
                label, prob = labels[i], float(p[i])
                if n_hands < need_hands.get(label, 1):
                    label, prob = NONE, 1.0                  # двуручный жест, а в кадре одна рука

            key = trigger.update(label, prob, now)
            if key and state["armed"]:
                kb.press(key)
                log.appendleft(f"{key.upper()}  ({trigger.current})")

            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            draw(frame, res)
            h, w = frame.shape[:2]
            status = ("DRY-RUN " if kb.dry_run else "") + ("ON" if state["armed"] else "PAUSED")
            color = (0, 220, 0) if state["armed"] else (0, 165, 255)
            text(frame, [f"{status}   {fps:.0f} fps", f"{label}  {prob:.2f}"], color)
            text(frame, list(log), (255, 255, 255), x=w - 170, y0=28, scale=0.6)
            cv2.rectangle(frame, (10, h - 18), (10 + int((w - 20) * prob), h - 8), color, -1)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("p"):
                toggle()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()