"""
3 Laboratorinis darbas - Vaizdu klasifikavimas konvoliuciniu neuroniniu tinklu (2D).
Duomenu rinkinys: Rock-Paper-Scissors (drgfreeman / Kaggle), 2188 PNG, 3 klases.

Vykdoma:
    python main.py

Rezultatai (mokymo kreives, summary.csv, geriausio modelio testavimo metrikos,
confusion matrix, ~30 testavimo pavyzdziu su prognozemis) - issaugomi i
3Lab/rezultatai/images/.
"""
from __future__ import annotations

import csv
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

# Mazinam TensorFlow trigsma logi prie jam uzkraunant.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")  # nereikia X displejaus, raso PNG i diska
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers

# -----------------------------------------------------------------------------
# Konfiguracija (kataloginiai keliai, atsitiktinumo seed)
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "rockpaperscissors"
RESULTS_DIR = THIS_DIR / "rezultatai" / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["paper", "rock", "scissors"]  # alfabetine tvarka, kaip ir aplankuose
IMG_SIZE = (128, 128)  # H x W - sumazinta is originalo (300x200), kad tilptu i atminti
NUM_CLASSES = len(CLASS_NAMES)
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


# -----------------------------------------------------------------------------
# Duomenu paruosimas
# -----------------------------------------------------------------------------
def load_images() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Iskraunam visus PNG i atminti, normalizuojam i [0,1].

    Grazinam: X (N,H,W,3) float32, y (N,) int, paths (N,) string sarasas
    (paths reikalingas pavyzdziu vaizdavimui.)
    """
    X, y, paths = [], [], []
    for label_idx, cname in enumerate(CLASS_NAMES):
        cdir = DATA_DIR / cname
        files = sorted(cdir.glob("*.png"))
        for fp in files:
            img = tf.keras.utils.load_img(str(fp), target_size=IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img) / 255.0  # float32 [0,1]
            X.append(arr)
            y.append(label_idx)
            paths.append(str(fp))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y, paths


def make_splits(X: np.ndarray, y: np.ndarray, paths: list[str]):
    """80/10/10 stratifikuotas suskirstymas - viena pastovi atsitiktinumo seed.

    Pastaba: PDF 1 punktas reikalauja apjungti is anksto suskirstytus duomenis ir
    perskirstyti is naujo. Ciaduomenu rinkinys jau yra vientisas (be train/val/test
    aplanku), todel tiesiog vykdom viena 80:10:10 stratifikuota suskirstyma.
    """
    paths = np.asarray(paths)
    # 80 train / 20 likes
    X_tr, X_rest, y_tr, y_rest, p_tr, p_rest = train_test_split(
        X, y, paths, test_size=0.20, stratify=y, random_state=SEED
    )
    # is likusio 20% padalinam per puse -> 10 val / 10 test
    X_va, X_te, y_va, y_te, p_va, p_te = train_test_split(
        X_rest, y_rest, p_rest, test_size=0.50, stratify=y_rest, random_state=SEED
    )
    return (X_tr, y_tr, p_tr), (X_va, y_va, p_va), (X_te, y_te, p_te)


# -----------------------------------------------------------------------------
# Modelio konstravimas pagal konfiguracija (laisvas architekturos keitimas)
# -----------------------------------------------------------------------------
def _make_activation_layer(name: str):
    """Grazinam keras sluoksni atitinkanti aktyvacija. LeakyReLU - atskiras layer."""
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


def build_model(cfg: dict, input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """Sukuriam Sequential modeli pagal cfg dictiona.

    cfg pavyzdys:
      {
        "blocks": [{"filters":32,"kernel":3,"pool":2,"dropout":0.0,"batch_norm":False,"activation":"relu"}, ...],
        "dense": [128],
        "dense_dropout": 0.0,
        "dense_batch_norm": False,
        "dense_activation": "relu",
        "optimizer": "adam", "learning_rate": 1e-3,
        "loss": "sparse_categorical_crossentropy",
      }
    """
    model = models.Sequential(name=cfg["name"].replace(".", "_"))
    model.add(layers.Input(shape=input_shape))

    for i, blk in enumerate(cfg["blocks"]):
        # Conv -> (BN) -> aktyvacija -> Pool -> (Dropout)
        # use_bias=False kai BN, nes BN turi savo bias.
        use_bias = not blk.get("batch_norm", False)
        model.add(
            layers.Conv2D(
                filters=blk["filters"],
                kernel_size=blk["kernel"],
                padding="same",
                use_bias=use_bias,
                name=f"conv2d_{i}",
            )
        )
        if blk.get("batch_norm", False):
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(_make_activation_layer(blk["activation"]))
        if blk.get("pool", 0):
            model.add(layers.MaxPooling2D(pool_size=blk["pool"], name=f"pool_{i}"))
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
# Eksperimento vykdymas + grafikai
# -----------------------------------------------------------------------------
def plot_history(history, title: str, save_path: Path):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, h["loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="val")
    axes[0].set_title("Paklaida (loss)")
    axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].plot(epochs, h["accuracy"], label="train")
    axes[1].plot(epochs, h["val_accuracy"], label="val")
    axes[1].set_title("Tikslumas (accuracy)")
    axes[1].set_xlabel("Epocha")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def append_summary(row: dict):
    new_file = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_experiment(cfg: dict, splits, input_shape, num_classes):
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = splits
    print(f"\n=== Eksperimentas: {cfg['name']} ===")
    print({k: v for k, v in cfg.items() if k != "blocks"})
    print("blocks =", cfg["blocks"])

    tf.keras.utils.set_random_seed(SEED)  # nors butu palyginama tarp bandymu
    model = build_model(cfg, input_shape, num_classes)
    model.summary(print_fn=lambda s: print("  " + s))

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=2,
        shuffle=True,
    )

    # Galutines mokymo, validavimo ir testavimo metrikos:
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
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "optimizer": cfg["optimizer"],
        "learning_rate": cfg["learning_rate"],
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
# Eksperimentu rinkinys (architekturu palyginimas + hiperparametru tyrimai)
# -----------------------------------------------------------------------------
def base_block(filters, *, kernel=3, pool=2, dropout=0.0, batch_norm=False, activation="relu"):
    return {"filters": filters, "kernel": kernel, "pool": pool,
            "dropout": dropout, "batch_norm": batch_norm, "activation": activation}


def base_cfg(name, blocks, dense, *, optimizer="adam", lr=1e-3,
             dense_dropout=0.0, dense_bn=False, dense_act="relu",
             epochs=20, batch_size=32, loss="sparse_categorical_crossentropy"):
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
    """Trys skirtingo gylio architekturos (PDF reikalauja >= 3)."""
    return [
        base_cfg(
            "arch_small",
            blocks=[base_block(32), base_block(64)],
            dense=[64], epochs=epochs,
        ),
        base_cfg(
            "arch_medium",
            blocks=[base_block(32), base_block(64), base_block(128)],
            dense=[128], epochs=epochs,
        ),
        base_cfg(
            "arch_large",
            blocks=[base_block(32), base_block(64), base_block(128), base_block(128)],
            dense=[256, 128], epochs=epochs,
        ),
    ]


def hp_configs_for(best_cfg, epochs):
    """Hiperparametru tyrimai - vykdomi ant geriausios architekturos.

    Aprepia: dropout, batch normalization, aktyvacijos funkcija, optimizatorius.
    """
    cfgs = []

    # Dropout sweep (vienodas dropout visuose conv ir dense sluoksniuose).
    for p in [0.0, 0.25, 0.5]:
        c = deepcopy(best_cfg)
        c["name"] = f"dropout_p{p}"
        for b in c["blocks"]:
            b["dropout"] = p
        c["dense_dropout"] = p
        c["epochs"] = epochs
        cfgs.append(c)
    # Tik dense dropout (atvejis: kelis sluoksnius su skirtingais nustatymais)
    c = deepcopy(best_cfg)
    c["name"] = "dropout_dense_only_0.5"
    for b in c["blocks"]:
        b["dropout"] = 0.0
    c["dense_dropout"] = 0.5
    c["epochs"] = epochs
    cfgs.append(c)

    # Batch normalization on/off
    for use_bn in [False, True]:
        c = deepcopy(best_cfg)
        c["name"] = f"bn_{'on' if use_bn else 'off'}"
        for b in c["blocks"]:
            b["batch_norm"] = use_bn
        c["dense_batch_norm"] = use_bn
        c["epochs"] = epochs
        cfgs.append(c)

    # Aktyvacijos funkcijos
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
# Geriausio modelio testavimas + confusion matrix + 30 pavyzdziu
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, classes, save_path: Path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Prognoze")
    ax.set_ylabel("Tikra klase")
    ax.set_title(title)
    # Rasom skaicius
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_best(best, splits, classes):
    (_, _, _), (_, _, _), (X_te, y_te, p_te) = splits
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

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(classes))))
    plot_confusion_matrix(cm, classes, RESULTS_DIR / "best_confusion_matrix.png",
                          title=f"Confusion matrix - {name}")
    # CSV variantas (lengva idet i ataskaita)
    with (RESULTS_DIR / "best_confusion_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + classes)
        for i, cname in enumerate(classes):
            w.writerow([cname] + [int(v) for v in cm[i]])

    # ~30 testavimo pavyzdziu (10 is kiekvienos klases)
    rng = np.random.default_rng(SEED)
    chosen_idx = []
    for c in range(len(classes)):
        idxs = np.where(y_te == c)[0]
        n_take = min(10, len(idxs))
        chosen_idx.extend(rng.choice(idxs, size=n_take, replace=False).tolist())
    chosen_idx = np.array(chosen_idx, dtype=int)

    sample_preds = np.argmax(model.predict(X_te[chosen_idx], verbose=0), axis=1)
    with (RESULTS_DIR / "sample_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index_in_test", "image_path", "true_label", "predicted_label", "correct"])
        for k, idx in enumerate(chosen_idx):
            tl = classes[int(y_te[idx])]
            pl = classes[int(sample_preds[k])]
            w.writerow([int(idx), p_te[idx], tl, pl, int(tl == pl)])

    # Vizualizuojam tinkleli (5 stulpeliai x 6 eilutes)
    n = len(chosen_idx)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.axis("off")
        if k < n:
            idx = chosen_idx[k]
            ax.imshow(X_te[idx])
            true_l = classes[int(y_te[idx])]
            pred_l = classes[int(sample_preds[k])]
            color = "green" if true_l == pred_l else "red"
            ax.set_title(f"T:{true_l}\nP:{pred_l}", fontsize=8, color=color)
    fig.suptitle(f"Testavimo pavyzdziai - {name} (T=tikra, P=prognoze)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_predictions_grid.png", dpi=140)
    plt.close(fig)

    print(f"Pavyzdziu rinkinio dydis: {len(chosen_idx)} (po {len(chosen_idx)//len(classes)} is klases)")
    print(f"Visi rezultatai: {RESULTS_DIR}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def _load_existing_summary():
    """Iskraunam jau atliktus eksperimentus is summary.csv (jei yra), kad galetume testi.

    Grazinam dict {name: row_dict}.
    """
    out = {}
    if SUMMARY_CSV.exists():
        with SUMMARY_CSV.open() as f:
            for row in csv.DictReader(f):
                out[row["name"]] = row
    return out


def _row_to_result(row):
    """Sustatom result-stiliaus dict is summary.csv eilutes (be model objekto)."""
    return {
        "name": row["name"],
        "model": None,  # bus reikalingas tik geriausiajam - persimokom is naujo
        "history": None,
        "train_acc": float(row["train_acc"]),
        "train_loss": float(row["train_loss"]),
        "val_acc": float(row["val_acc"]),
        "val_loss": float(row["val_loss"]),
        "test_acc": float(row["test_acc"]),
        "test_loss": float(row["test_loss"]),
        "cfg": None,
    }


def _maybe_run(cfg, splits, input_shape, num_classes, done_rows):
    """Vykdom eksperimenta tik jei jis dar nepadarytas (rezumavimo logika)."""
    if cfg["name"] in done_rows:
        print(f"\n[SKIP] {cfg['name']} jau yra summary.csv - praleidziam.")
        r = _row_to_result(done_rows[cfg["name"]])
        r["cfg"] = cfg
        return r
    return run_experiment(cfg, splits, input_shape, num_classes)


def main():
    print("Krauni vaizdus is", DATA_DIR)
    X, y, paths = load_images()
    print(f"Vaizdu kiekis: {len(X)}, dydis: {X.shape[1:]}, klases: {CLASS_NAMES}")
    for c, cname in enumerate(CLASS_NAMES):
        print(f"  {cname}: {(y == c).sum()}")

    splits = make_splits(X, y, paths)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = splits
    print(f"Splits: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # Galima keisti epochu skaiciu is komandines eilutes: python main.py 30
    # Antras argumentas "fresh" istrina sena summary.csv ir pradeda nuo nulio.
    epochs_default = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    fresh = (len(sys.argv) > 2 and sys.argv[2].lower() == "fresh")
    if fresh and SUMMARY_CSV.exists():
        print("[FRESH] istrinam sena summary.csv")
        SUMMARY_CSV.unlink()
    done_rows = _load_existing_summary()
    if done_rows:
        print(f"Resume: {len(done_rows)} eksperimentai jau atlikti, juos praleisim.")

    input_shape = X.shape[1:]
    results: list[dict] = []

    # ---------- A. Architekturu palyginimas ----------
    print("\n\n##### A. ARCHITEKTURU PALYGINIMAS #####")
    for cfg in architecture_configs(epochs_default):
        results.append(_maybe_run(cfg, splits, input_shape, NUM_CLASSES, done_rows))

    # Geriausia (pagal validavimo tiksluma) - jos pagrindu tesiam hiperparametru tyrima
    arch_results = sorted(results, key=lambda r: r["val_acc"], reverse=True)
    best_arch = arch_results[0]
    print(f"\nGeriausia architektura pagal val_acc: {best_arch['name']} "
          f"(val_acc={best_arch['val_acc']:.4f})")
    # Jei geriausioji architektura buvo prakraustyta is CSV (be cfg), atkuriam ja is konfiguraciju saraso.
    if best_arch.get("cfg") is None:
        for c in architecture_configs(epochs_default):
            if c["name"] == best_arch["name"]:
                best_arch_cfg = deepcopy(c)
                break
    else:
        best_arch_cfg = deepcopy(best_arch["cfg"])

    # ---------- B-E. Hiperparametru tyrimai ----------
    print("\n\n##### B-E. HIPERPARAMETRU TYRIMAI (su geriausia architektura) #####")
    for cfg in hp_configs_for(best_arch_cfg, epochs_default):
        results.append(_maybe_run(cfg, splits, input_shape, NUM_CLASSES, done_rows))

    # ---------- Geriausias bandymas (didz. mokymo tikslumas, maz. mokymo paklaida) ----
    # PDF 4 punktas: didziausias klasifikavimo tikslumas ir mazausia paklaida MOKYMO duomenims.
    best = max(results, key=lambda r: (r["train_acc"], -r["train_loss"]))
    print(f"\nIs viso atlikta {len(results)} eksperimentu.")
    print(f"Geriausias eksperimentas mokymo aibeje: {best['name']} "
          f"(train_acc={best['train_acc']:.4f}, train_loss={best['train_loss']:.4f})")

    # Jei geriausiajam neturim modelio (buvo praleistas), persimokom ji is naujo.
    if best["model"] is None:
        print(f"[REFIT] {best['name']} buvo praleistas - persimokom modeli is naujo testavimui.")
        # Rasom konfiguracija is architektura/HP saraso.
        all_cfgs = architecture_configs(epochs_default) + hp_configs_for(best_arch_cfg, epochs_default)
        cfg = next((c for c in all_cfgs if c["name"] == best["name"]), None)
        assert cfg is not None, f"Negaliu rasti cfg pavadinimu {best['name']}"
        # Persimokom be summary papildymo: vykdom tiesiogiai, kad gautume model objekta.
        tf.keras.utils.set_random_seed(SEED)
        model = build_model(cfg, input_shape, NUM_CLASSES)
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                  epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=2, shuffle=True)
        best["model"] = model

    evaluate_best(best, splits, CLASS_NAMES)

    print("\nVisas santraukos failas:", SUMMARY_CSV)
    print("Saltinai:")
    print("  Rock-Paper-Scissors: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/data")


if __name__ == "__main__":
    main()
