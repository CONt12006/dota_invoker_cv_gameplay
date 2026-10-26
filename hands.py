"""Руки с вебки -> вектор признаков. Общий код для записи, обучения и контроллера.

Признаки кадра (156 чисел):
  форма левой кисти      63  — точки относительно запястья, масштаб по размеру ладони
  форма правой кисти     63
  правое запястье относительно левого   3  — где руки друг относительно друга
  расстояния между кончиками пальцев рук 25  — «пальцы касаются» (жесты двумя руками)
  флаги «рука найдена»   2
Форма кисти не зависит от того, где рука в кадре и как далеко от камеры, но зависит от поворота
(ладонь / ребро / тыльная сторона) — это важно для различения поз.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dota_gestures"                  # записанные жесты
MODEL_PATH = ROOT / "models" / "dota_gestures.joblib"
CONFIG_PATH = Path(__file__).with_name("config.json")

NONE = "none"
TIPS = [4, 8, 12, 16, 20]          # кончики: большой, указательный, средний, безымянный, мизинец
FEAT_DIM = 63 + 63 + 3 + 25 + 2
FEATURE_VERSION = 1

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def gesture_names(config=None):
    config = config or load_config()
    return [NONE] + list(config["gestures"])


def create_hands():
    # model_complexity=0 — самая быстрая модель: для управления игрой задержка важнее точности
    return mp_hands.Hands(max_num_hands=2, model_complexity=0,
                          min_detection_confidence=0.6, min_tracking_confidence=0.5)


def open_camera(index=0, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть камеру {index}")
    return cap


def read_frame(cap):
    """Кадр сразу зеркалим (как в зеркале) — тогда MediaPipe правильно определяет левую/правую руку."""
    ok, frame = cap.read()
    return (cv2.flip(frame, 1) if ok else None)


def detect(hands, frame_bgr):
    """-> (left (21,3) | None, right (21,3) | None, raw results)"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    left = right = None
    found = []
    if res.multi_hand_landmarks:
        for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32)
            found.append((pts, handed.classification[0].label))
    if len(found) >= 2:
        # Две руки: распределяем по положению в кадре (кадр зеркальный — левая рука слева).
        # Метка MediaPipe при сложенных/касающихся руках («домик», ладони вместе) часто
        # даёт обеим рукам одну и ту же сторону — по координате надёжнее.
        a, b = sorted(found[:2], key=lambda h: float(h[0][0, 0]))
        left, right = a[0], b[0]
    elif found:
        pts, label = found[0]
        if label == "Left":
            left = pts
        else:
            right = pts
    return left, right, res


def _local(pts):
    rel = pts - pts[0]
    scale = float(np.linalg.norm(rel[9, :2])) + 1e-6   # запястье -> основание среднего пальца
    return rel / scale, scale


def features(left, right):
    f = np.zeros(FEAT_DIM, np.float32)
    scales = []
    if left is not None:
        loc, s = _local(left)
        f[0:63] = loc.ravel()
        scales.append(s)
    if right is not None:
        loc, s = _local(right)
        f[63:126] = loc.ravel()
        scales.append(s)
    if left is not None and right is not None:
        scale = float(np.mean(scales))
        f[126:129] = (right[0] - left[0]) / scale
        d = np.linalg.norm(left[TIPS][:, None, :2] - right[TIPS][None, :, :2], axis=-1) / scale
        f[129:154] = d.ravel()
    f[154] = left is not None
    f[155] = right is not None
    return f


def augment(X, y, rng, copies=4):
    """Делает из каждого кадра несколько чуть изменённых: шум, масштаб, небольшой поворот кистей.
    Возвращает (X_aug, y_aug)."""
    out, labels = [X], [y]
    for _ in range(copies):
        A = X.copy()
        n = len(A)
        for sl, flag in ((slice(0, 63), 154), (slice(63, 126), 155)):
            pts = A[:, sl].reshape(n, 21, 3)
            ang = np.deg2rad(rng.uniform(-10, 10, n))
            c, s = np.cos(ang), np.sin(ang)
            px, py = pts[..., 0].copy(), pts[..., 1].copy()
            pts[..., 0] = c[:, None] * px - s[:, None] * py
            pts[..., 1] = s[:, None] * px + c[:, None] * py
            pts *= rng.uniform(0.92, 1.08, (n, 1, 1))
            pts += rng.normal(0, 0.03, pts.shape) * A[:, flag][:, None, None]
            A[:, sl] = pts.reshape(n, 63)
        both = (A[:, 154] * A[:, 155])[:, None]
        A[:, 126:154] *= rng.uniform(0.9, 1.1, (n, 1))
        A[:, 126:154] += rng.normal(0, 0.05, (n, 28)) * both
        out.append(A)
        labels.append(y)
    # Одна рука в кадре: MediaPipe иногда путает, левая она или правая, и признаки попадают
    # «не в ту половину». Учим модель узнавать однорукий жест в обеих половинах.
    one = (X[:, 154] + X[:, 155]) == 1
    if one.any():
        S = X[one].copy()
        S[:, 0:63], S[:, 63:126] = X[one][:, 63:126], X[one][:, 0:63]
        S[:, 154], S[:, 155] = X[one][:, 155], X[one][:, 154]
        out.append(S)
        labels.append(y[one])

        # Зеркалим однорукие кадры: правая рука -> левая и наоборот.
        # Отражение по горизонтали = смена знака x у точек кисти.
        M = X[one].copy()
        for sl in (slice(0, 63), slice(63, 126)):
            pts = M[:, sl].reshape(-1, 21, 3)
            pts[..., 0] *= -1
            M[:, sl] = pts.reshape(-1, 63)
        M2 = M.copy()
        M2[:, 0:63], M2[:, 63:126] = M[:, 63:126], M[:, 0:63]
        M2[:, 154], M2[:, 155] = M[:, 155], M[:, 154]
        out += [M, M2]
        labels += [y[one], y[one]]
    return np.concatenate(out), np.concatenate(labels)


def draw(frame, res):
    if res.multi_hand_landmarks:
        for lms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                                   mp_styles.get_default_hand_landmarks_style(),
                                   mp_styles.get_default_hand_connections_style())


def text(frame, lines, color=(0, 255, 0), x=10, y0=28, scale=0.65):
    for i, t in enumerate(lines):
        y = y0 + i * int(30 * scale / 0.65)
        cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)