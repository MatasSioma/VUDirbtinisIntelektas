"""
3 Laboratorinis darbas - Laiko eiluciu klasifikavimas konvoliuciniu neuroniniu tinklu (1D).
Duomenu rinkinys: CMU Keystroke Dynamics (slaptazodis "tie5Roanl"),
51 dalyvis x 8 sesijos x 50 pakartojimu, 31 laiko poziniai.

Siame skripte:
- pasirenkam pirmus 30 dalyviu (sortuotai is 51) ir 1 sesija -> 1500 irasu, 30 klasiu;
- 80:10:10 stratifikuotas suskirstymas;
- 1D konvoliucinis tinklas, eksperimentai su >= 3 architekturomis ir hiperparametrais.

Vykdoma:
    python main_ts.py
"""
from __future__ import annotations

import csv
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers

# -----------------------------------------------------------------------------
# Konfiguracija
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

THIS_DIR = Path(__file__).resolve().parent
CSV_PATH = THIS_DIR / "DSL-StrongPasswordData.csv"
RESULTS_DIR = THIS_DIR / "rezultatai" / "timeseries"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_USERS = 30          # PDF reikalavimas: 30 klasiu
SESSION = 1             # PDF: pasirinkti viena sesija
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


# -----------------------------------------------------------------------------
# Duomenu paruosimas
# -----------------------------------------------------------------------------
def load_dataset():
    """Iskraunam CSV, atsirenkam 30 dalyviu ir 1 sesija, padarom (N, 31, 1) tensoru.

    Grazinam: X (N, 31, 1) float32, y (N,) int (0..29), class_names (30 stringu),
    feature_names (31 stringu).
    """
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in ("subject", "sessionIndex", "rep")]
    assert len(feature_cols) == 31, f"Tikektasi 31 pozimio, gauta {len(feature_cols)}"

    df = df[df["sessionIndex"] == SESSION].copy()
    # Pirmi 30 unikaliu dalyviu (sorted, kad butu deterministiska)
    subjects = sorted(df["subject"].unique())[:NUM_USERS]
    df = df[df["subject"].isin(subjects)].reset_index(drop=True)

    le = LabelEncoder().fit(subjects)
    y = le.transform(df["subject"].values).astype(np.int64)
    X = df[feature_cols].values.astype(np.float32)  # (N, 31)
    X = X[..., np.newaxis]  # -> (N, 31, 1) - reikalingas Conv1D
    return X, y, list(le.classes_), feature_cols


def make_splits(X, y):
    """80/10/10 stratifikuotas; standartizuojam pagal mokymo aibes statistikas."""
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED
    )
    # Standartizuojam pagal stulpelius (per pozini), naudodami tik mokymo aibes statistikas.
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std
    X_te = (X_te - mean) / std
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


# -----------------------------------------------------------------------------
# Modelio konstravimas
# -----------------------------------------------------------------------------
def _make_activation_layer(name: str):
    name = name.lower()
    if name == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    return layers.Activation(name)


def _make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    if name == "adamw":
        return optimizers.AdamW(learning_rate=lr)
    raise ValueError(f"Nezinomas optimizatorius: {name}")


def build_model(cfg: dict, input_shape, num_classes):
    """Sequential 1D CNN pagal cfg (analogiska kaip image versijoje)."""
    model = models.Sequential(name=cfg["name"].replace(".", "_"))
    model.add(layers.Input(shape=input_shape))

    for i, blk in enumerate(cfg["blocks"]):
        use_bias = not blk.get("batch_norm", False)
        model.add(layers.Conv1D(
            filters=blk["filters"],
            kernel_size=blk["kernel"],
            padding="same",
            use_bias=use_bias,
            name=f"conv1d_{i}",
        ))
        if blk.get("batch_norm", False):
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(_make_activation_layer(blk["activation"]))
        if blk.get("pool", 0):
            # Po keliu poolingu eilute trumpeja, todel apsidraudziam pool=1 jei per trumpa.
            model.add(layers.MaxPooling1D(pool_size=blk["pool"], padding="same",
                                          name=f"pool_{i}"))
        if blk.get("dropout", 0.0) > 0.0:
            model.add(layers.Dropout(blk["dropout"], name=f"drop_{i}"))

    model.add(layers.Flatten())

    for j, units in enumerate(cfg.get("dense", [])):
        model.add(layers.Dense(units, use_bias=not cfg.get("dense_batch_norm", False),
                               name=f"dense_{j}"))
        if cfg.get("dense_batch_norm", False):
            model.add(layers.BatchNormalization(name=f"dense_bn_{j}"))
        model.add(_make_activation_layer(cfg.get("dense_activation", "relu")))
        if cfg.get("dense_dropout", 0.0) > 0.0:
            model.add(layers.Dropout(cfg["dense_dropout"], name=f"dense_drop_{j}"))

    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=_make_optimizer(cfg.get("optimizer", "adam"),
                                  cfg.get("learning_rate", 1e-3)),
        loss=cfg.get("loss", "sparse_categorical_crossentropy"),
        metrics=["accuracy"],
    )
    return model


# -----------------------------------------------------------------------------
# Eksperimento helperiai (tie patys kaip image versijoje, tik be path-related)
# -----------------------------------------------------------------------------
def plot_history(history, title, save_path):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, h["loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="val")
    axes[0].set_title("Paklaida (loss)"); axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].plot(epochs, h["accuracy"], label="train")
    axes[1].plot(epochs, h["val_accuracy"], label="val")
    axes[1].set_title("Tikslumas (accuracy)"); axes[1].set_xlabel("Epocha")
    axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close(fig)


def append_summary(row):
    new_file = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_experiment(cfg, splits, input_shape, num_classes):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    print(f"\n=== Eksperimentas: {cfg['name']} ===")
    print({k: v for k, v in cfg.items() if k != "blocks"})
    print("blocks =", cfg["blocks"])

    tf.keras.utils.set_random_seed(SEED)
    model = build_model(cfg, input_shape, num_classes)
    model.summary(print_fn=lambda s: print("  " + s))

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=2, shuffle=True,
    )

    train_loss, train_acc = model.evaluate(X_tr, y_tr, verbose=0, batch_size=cfg["batch_size"])
    val_loss, val_acc = model.evaluate(X_va, y_va, verbose=0, batch_size=cfg["batch_size"])
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0, batch_size=cfg["batch_size"])
    print(f"[{cfg['name']}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
          f"test_acc={test_acc:.4f} | train_loss={train_loss:.4f} "
          f"val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    plot_history(history, f"{cfg['name']} - mokymo kreives",
                 RESULTS_DIR / f"{cfg['name']}_curves.png")

    append_summary({
        "name": cfg["name"],
        "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
        "optimizer": cfg["optimizer"], "learning_rate": cfg["learning_rate"],
        "n_conv_blocks": len(cfg["blocks"]),
        "filters": "-".join(str(b["filters"]) for b in cfg["blocks"]),
        "kernel": cfg["blocks"][0]["kernel"],
        "pool": cfg["blocks"][0]["pool"],
        "conv_dropout": cfg["blocks"][0].get("dropout", 0.0),
        "dense": "-".join(str(d) for d in cfg.get("dense", [])),
        "dense_dropout": cfg.get("dense_dropout", 0.0),
        "batch_norm": cfg["blocks"][0].get("batch_norm", False),
        "activation": cfg["blocks"][0]["activation"],
        "loss_fn": cfg["loss"],
        "train_loss": round(float(train_loss), 5),
        "train_acc": round(float(train_acc), 5),
        "val_loss": round(float(val_loss), 5),
        "val_acc": round(float(val_acc), 5),
        "test_loss": round(float(test_loss), 5),
        "test_acc": round(float(test_acc), 5),
    })

    return {
        "name": cfg["name"], "model": model, "history": history,
        "train_acc": train_acc, "train_loss": train_loss,
        "val_acc": val_acc, "val_loss": val_loss,
        "test_acc": test_acc, "test_loss": test_loss,
        "cfg": cfg,
    }


# -----------------------------------------------------------------------------
# Eksperimentu rinkinys
# -----------------------------------------------------------------------------
def base_block(filters, *, kernel=3, pool=2, dropout=0.0, batch_norm=False, activation="relu"):
    return {"filters": filters, "kernel": kernel, "pool": pool,
            "dropout": dropout, "batch_norm": batch_norm, "activation": activation}


def base_cfg(name, blocks, dense, *, optimizer="adam", lr=1e-3,
             dense_dropout=0.0, dense_bn=False, dense_act="relu",
             epochs=25, batch_size=32, loss="sparse_categorical_crossentropy"):
    return {
        "name": name,
        "blocks": deepcopy(blocks),
        "dense": list(dense),
        "dense_dropout": dense_dropout,
        "dense_batch_norm": dense_bn,
        "dense_activation": dense_act,
        "optimizer": optimizer,
        "learning_rate": lr,
        "loss": loss,
        "epochs": epochs,
        "batch_size": batch_size,
    }


def architecture_configs(epochs):
    """3 skirtingos architekturos. Eilutes ilgis = 31, todel naudojam pool=2 atsargiai."""
    return [
        base_cfg(
            "arch_small",
            blocks=[base_block(32, kernel=3, pool=2),
                    base_block(64, kernel=3, pool=2)],
            dense=[64], epochs=epochs,
        ),
        base_cfg(
            "arch_medium",
            blocks=[base_block(32, kernel=5, pool=2),
                    base_block(64, kernel=3, pool=2),
                    base_block(128, kernel=3, pool=1)],  # pool=1 - issaugom likusi ilgi
            dense=[128], epochs=epochs,
        ),
        base_cfg(
            "arch_large",
            blocks=[base_block(64, kernel=5, pool=2),
                    base_block(128, kernel=3, pool=2),
                    base_block(128, kernel=3, pool=1),
                    base_block(64, kernel=3, pool=1)],
            dense=[256, 128], epochs=epochs,
        ),
    ]


def hp_configs_for(best_cfg, epochs):
    cfgs = []
    # Dropout sweep
    for p in [0.0, 0.25, 0.5]:
        c = deepcopy(best_cfg)
        c["name"] = f"dropout_p{p}"
        for b in c["blocks"]:
            b["dropout"] = p
        c["dense_dropout"] = p
        c["epochs"] = epochs
        cfgs.append(c)
    # Tik dense dropout
    c = deepcopy(best_cfg)
    c["name"] = "dropout_dense_only_0.5"
    for b in c["blocks"]:
        b["dropout"] = 0.0
    c["dense_dropout"] = 0.5
    c["epochs"] = epochs
    cfgs.append(c)
    # Batch norm
    for use_bn in [False, True]:
        c = deepcopy(best_cfg)
        c["name"] = f"bn_{'on' if use_bn else 'off'}"
        for b in c["blocks"]:
            b["batch_norm"] = use_bn
        c["dense_batch_norm"] = use_bn
        c["epochs"] = epochs
        cfgs.append(c)
    # Aktyvacijos
    for act in ["relu", "tanh", "elu", "leaky_relu"]:
        c = deepcopy(best_cfg)
        c["name"] = f"act_{act}"
        for b in c["blocks"]:
            b["activation"] = act
        c["dense_activation"] = act
        c["epochs"] = epochs
        cfgs.append(c)
    # Optimizatoriai
    for opt in ["adam", "sgd", "rmsprop", "adamw"]:
        c = deepcopy(best_cfg)
        c["name"] = f"opt_{opt}"
        c["optimizer"] = opt
        c["epochs"] = epochs
        cfgs.append(c)
    return cfgs


# -----------------------------------------------------------------------------
# Geriausio modelio testavimas + confusion matrix + ~30 pavyzdziu
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, save_path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Prognoze")
    ax.set_ylabel("Tikra klase")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_best(best, splits, classes):
    (_, _), (_, _), (X_te, y_te) = splits
    model = best["model"]
    name = best["name"]

    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n*** Geriausias modelis: {name} ***")
    print(f"Testavimo paklaida (loss): {test_loss:.4f}")
    print(f"Testavimo tikslumas (acc): {test_acc:.4f}")

    (RESULTS_DIR / "best_test_metrics.txt").write_text(
        f"best_model={name}\n"
        f"test_loss={test_loss:.5f}\n"
        f"test_accuracy={test_acc:.5f}\n"
        f"train_acc={best['train_acc']:.5f}\n"
        f"train_loss={best['train_loss']:.5f}\n"
    )

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(classes))))
    plot_confusion_matrix(cm, classes, RESULTS_DIR / "best_confusion_matrix.png",
                          title=f"Confusion matrix - {name}")
    with (RESULTS_DIR / "best_confusion_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + classes)
        for i, cname in enumerate(classes):
            w.writerow([cname] + [int(v) for v in cm[i]])

    # Po viena pavyzdi is kiekvienos klases (30 klasiu -> 30 irasu)
    rng = np.random.default_rng(SEED)
    chosen_idx = []
    for c in range(len(classes)):
        idxs = np.where(y_te == c)[0]
        if len(idxs) > 0:
            chosen_idx.append(int(rng.choice(idxs)))
    chosen_idx = np.array(chosen_idx, dtype=int)
    sample_preds = np.argmax(model.predict(X_te[chosen_idx], verbose=0), axis=1)

    with (RESULTS_DIR / "sample_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index_in_test", "true_subject", "predicted_subject", "correct"])
        for k, idx in enumerate(chosen_idx):
            tl = classes[int(y_te[idx])]
            pl = classes[int(sample_preds[k])]
            w.writerow([int(idx), tl, pl, int(tl == pl)])

    correct = int(sum(int(y_te[idx]) == int(sample_preds[k]) for k, idx in enumerate(chosen_idx)))
    print(f"Pavyzdziu rinkinys: {len(chosen_idx)} (po 1 is klases), is ju teisingai: {correct}")
    print(f"Visi rezultatai: {RESULTS_DIR}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("Krauni laiko eilutes is", CSV_PATH)
    X, y, classes, feature_cols = load_dataset()
    print(f"Dalyviu (klasiu): {len(classes)}, irasu: {len(X)}, pozimiai: {X.shape[1]} (kanal: {X.shape[2]})")
    print(f"Sesija: {SESSION}, dalyviai: {classes}")

    splits = make_splits(X, y)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    print(f"Splits: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    epochs_default = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()

    input_shape = X.shape[1:]  # (31, 1)
    results = []

    print("\n\n##### A. ARCHITEKTURU PALYGINIMAS #####")
    for cfg in architecture_configs(epochs_default):
        results.append(run_experiment(cfg, splits, input_shape, len(classes)))

    arch_results = sorted(results, key=lambda r: r["val_acc"], reverse=True)
    best_arch = arch_results[0]
    print(f"\nGeriausia architektura pagal val_acc: {best_arch['name']} "
          f"(val_acc={best_arch['val_acc']:.4f})")
    best_arch_cfg = deepcopy(best_arch["cfg"])

    print("\n\n##### B-E. HIPERPARAMETRU TYRIMAI (su geriausia architektura) #####")
    for cfg in hp_configs_for(best_arch_cfg, epochs_default):
        results.append(run_experiment(cfg, splits, input_shape, len(classes)))

    best = max(results, key=lambda r: (r["train_acc"], -r["train_loss"]))
    print(f"\nIs viso atlikta {len(results)} eksperimentu.")
    print(f"Geriausias eksperimentas mokymo aibeje: {best['name']} "
          f"(train_acc={best['train_acc']:.4f}, train_loss={best['train_loss']:.4f})")

    evaluate_best(best, splits, classes)

    print("\nVisas santraukos failas:", SUMMARY_CSV)
    print("Saltinis: https://www.cs.cmu.edu/~keystroke/")


if __name__ == "__main__":
    main()
