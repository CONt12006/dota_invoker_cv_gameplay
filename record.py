"""Запись своих жестов с вебки.

python src/dota/record.py

Клавиши в окне:
  1..9      выбрать жест (0 — none, «нет жеста»)
  n / p     следующий / предыдущий жест
  SPACE     записать сессию: 1 с подготовки, затем --seconds секунд (по умолчанию 6)
  u         удалить последнюю сессию
  q / Esc   выход

Во время записи держи жест и слегка меняй его: поворачивай кисть, двигай ближе/дальше,
выше/ниже. Для «none» — опусти руки, почеши нос, держи руки расслабленно, показывай
переходы между жестами: это учит модель НЕ нажимать клавиши случайно.
Каждая сессия — файл data/dota_gestures/<жест>/<время>.npy (кадры × 156).
"""
import argparse
import time

import cv2
import numpy as np

from hands import (DATA_DIR, NONE, create_hands, detect, draw, features, gesture_names,
                   load_config, open_camera, read_frame, text)

PREP_SEC = 1.0


def main():
    config = load_config()
    names = gesture_names(config)
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=config.get("camera", 0))
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    def count(name):
        d = DATA_DIR / name
        return sum(len(np.load(f, mmap_mode="r")) for f in d.glob("*.npy")) if d.exists() else 0

    counts = {n: count(n) for n in names}
    idx, state, t0, buf, last = 1 if len(names) > 1 else 0, "idle", 0.0, [], None
    cap = open_camera(args.camera)
    win = "record gestures"
    with create_hands() as hands:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            left, right, res = detect(hands, frame)
            draw(frame, res)
            now, name = time.time(), names[idx]

            if state == "prep" and now >= t0:
                state, t0, buf = "rec", now, []
            if state == "rec":
                buf.append(features(left, right))
                if now - t0 >= args.seconds:
                    out = DATA_DIR / name
                    out.mkdir(parents=True, exist_ok=True)
                    last = out / f"{int(now * 1000)}.npy"
                    np.save(last, np.stack(buf))
                    counts[name] += len(buf)
                    print(f"{name}: +{len(buf)} кадров -> {last}")
                    state = "idle"

            h, w = frame.shape[:2]
            hint = config["gestures"].get(name, {}).get("hint", "none: hands down, random moves, transitions between gestures")
            key = config["gestures"].get(name, {}).get("key", "-")
            text(frame, [f"[{idx}] {name}  -> key '{key}'", f"frames: {counts[name]}",
                         "1-9/n/p: gesture  SPACE: record  u: undo  q: quit"])
            text(frame, [hint], (200, 200, 255), y0=h - 15, scale=0.5)
            if state == "prep":
                text(frame, [f"{t0 - now:.1f}"], (0, 200, 255), x=w // 2 - 30, y0=h // 2, scale=2.0)
            elif state == "rec":
                cv2.circle(frame, (w - 30, 30), 14, (0, 0, 255), -1)
                cv2.rectangle(frame, (0, h - 6), (int(w * (now - t0) / args.seconds), h), (0, 0, 255), -1)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if state != "idle":
                continue
            if ord("0") <= k <= ord("9") and k - ord("0") < len(names):
                idx = k - ord("0")
            elif k == ord("n"):
                idx = (idx + 1) % len(names)
            elif k == ord("p"):
                idx = (idx - 1) % len(names)
            elif k == ord(" "):
                state, t0 = "prep", now + PREP_SEC
            elif k == ord("u") and last is not None and last.exists():
                n = len(np.load(last))
                counts[last.parent.name] -= n
                last.unlink()
                print(f"удалено: {last}")
                last = None

    cap.release()
    cv2.destroyAllWindows()
    print("\nИтого кадров:", counts)
    few = [n for n, c in counts.items() if c < 300]
    if few:
        print("Мало данных (желательно 300+ кадров, 3+ сессии):", ", ".join(few))


if __name__ == "__main__":
    main()