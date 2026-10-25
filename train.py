"""Обучение классификатора жестов на записях из data/dota_gestures.

python src/dota/train.py

Модель — небольшая MLP-сеть (sklearn): жест определяется по одному кадру, обучение — секунды на CPU.
Качество проверяется на СЕССИЯХ, которых модель не видела (каждая запись — отдельная сессия),
иначе соседние почти одинаковые кадры дали бы завышенную точность.
"""
import argparse

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hands import DATA_DIR, FEAT_DIM, FEATURE_VERSION, MODEL_PATH, NONE, augment, gesture_names


def load(names):
    X, y, groups = [], [], []
    for ci, name in enumerate(names):
        for g, f in enumerate(sorted((DATA_DIR / name).glob("*.npy"))):
            a = np.load(f).astype(np.float32)
            if a.ndim != 2 or a.shape[1] != FEAT_DIM:
                print(f"пропускаю {f}: форма {a.shape}, нужна (*, {FEAT_DIM})")
                continue
            X.append(a)
            y.append(np.full(len(a), ci))
            groups.append(np.full(len(a), hash((name, f.name)) % (2 ** 31)))
    if not X:
        raise SystemExit(f"Нет записей в {DATA_DIR}. Сначала: python src/dota/record.py")
    return np.concatenate(X), np.concatenate(y), np.concatenate(groups)


def make_model(seed):
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
        max_iter=300, early_stopping=True, n_iter_no_change=15, random_state=seed))


def fit(X, y, seed):
    rng = np.random.default_rng(seed)
    Xa, ya = augment(X, y, rng)
    return make_model(seed).fit(Xa, ya)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_names = gesture_names()
    X, y, groups = load(all_names)
    present = sorted(set(y.tolist()))
    names = [all_names[i] for i in present]
    remap = {old: new for new, old in enumerate(present)}
    y = np.array([remap[v] for v in y])
    missing = [n for n in all_names if n not in names]
    if missing:
        print("ВНИМАНИЕ: нет записей для:", ", ".join(missing))
    if NONE not in names:
        print("ВНИМАНИЕ: нет класса none — модель будет нажимать клавиши на любое движение рук!")
    print("Кадров по жестам:", {n: int((y == i).sum()) for i, n in enumerate(names)})

    sessions = {n: len(set(groups[y == i])) for i, n in enumerate(names)}
    n_splits = min(3, min(sessions.values()))
    if n_splits >= 2:
        pred = np.empty_like(y)
        for tr, te in GroupKFold(n_splits).split(X, y, groups):
            model = fit(X[tr], y[tr], args.seed)
            pred[te] = model.predict(X[te])
        print(f"\nПроверка на незнакомых сессиях ({n_splits} фолда):")
        print(classification_report(y, pred, target_names=names, digits=3, zero_division=0))
        cm = confusion_matrix(y, pred)
        w = max(len(n) for n in names)
        print(" " * (w + 1) + " ".join(f"{n[:6]:>6}" for n in names))
        for n, row in zip(names, cm):
            print(f"{n:>{w}} " + " ".join(f"{v:6d}" for v in row))
    else:
        print("\nДля честной проверки нужно минимум 2 сессии на каждый жест — пропускаю проверку:", sessions)

    model = fit(X, y, args.seed)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": names, "feature_version": FEATURE_VERSION}, MODEL_PATH)
    print(f"\nМодель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()