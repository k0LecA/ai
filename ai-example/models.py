# ============================================================
# MODELIŲ TRENIRAVIMAS IR PALYGINIMAS – Cats vs Dogs
# ============================================================
# Prieš tai paleisk: 1_image_preprocessing.py
# Įkelia: X_train.npy, X_test.npy, y_train.npy, y_test.npy
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import time

# ─────────────────────────────────────────
# 1. DUOMENŲ ĮKĖLIMAS
# ─────────────────────────────────────────
print("Įkeliami duomenys...")
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────
# 2. MODELIŲ APIBRĖŽIMAS
# ─────────────────────────────────────────
modeliai = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,   # 100 medžių
        max_depth=20,
        random_state=42,
        n_jobs=-1           # naudoja visus CPU branduolius
    ),
    "SVM": SVC(
        kernel="rbf",       # radial basis function – geriausias vaizdams
        C=1.0,
        random_state=42
    ),
}

# ─────────────────────────────────────────
# 3. TRENIRAVIMAS IR VERTINIMAS
# ─────────────────────────────────────────
rezultatai = {}

print("\n" + "=" * 55)
print("MODELIŲ TRENIRAVIMAS")
print("=" * 55)

for pavadinimas, modelis in modeliai.items():
    print(f"\n{pavadinimas}...")

    # Treniravimas + laiko matavimas
    t0 = time.time()
    modelis.fit(X_train, y_train)
    treniravimo_laikas = time.time() - t0

    # Prognozės
    y_pred = modelis.predict(X_test)

    # Metrikos
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    rezultatai[pavadinimas] = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "cm":        cm,
        "y_pred":    y_pred,
        "laikas":    treniravimo_laikas,
    }

    print(f"  Laikas:    {treniravimo_laikas:.1f}s")
    print(f"  Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-score:  {f1:.3f}")

# ─────────────────────────────────────────
# 4. PALYGINIMO LENTELĖ
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("REZULTATŲ PALYGINIMAS")
print("=" * 55)
print(f"{'Modelis':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Laikas':>8}")
print("-" * 55)
for pav, r in rezultatai.items():
    print(f"{pav:<22} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
          f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['laikas']:>6.1f}s")

geriausias = max(rezultatai, key=lambda k: rezultatai[k]["f1"])
print(f"\nGeriausias pagal F1: {geriausias} "
      f"({rezultatai[geriausias]['f1']:.3f})")

# ─────────────────────────────────────────
# 5. GRAFIKAI
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Modelių palyginimas – Cats vs Dogs", fontsize=14, fontweight="bold")

# 5a–5c: Confusion Matrix kiekvienam modeliui
for i, (pav, r) in enumerate(rezultatai.items()):
    sns.heatmap(
        r["cm"], annot=True, fmt="d", ax=axes[0, i],
        cmap="Blues", cbar=False,
        xticklabels=["Katė", "Šuo"],
        yticklabels=["Katė", "Šuo"]
    )
    axes[0, i].set_title(f"{pav}\nAccuracy: {r['accuracy']:.1%}")
    axes[0, i].set_xlabel("Prognozuota")
    axes[0, i].set_ylabel("Tikroji")

# 5d: Metrikų palyginimas
metrikos = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metrikos))
plotis = 0.25
spalvos = ["#4C9BE8", "#E8754C", "#5DCA9E"]

for j, (pav, r) in enumerate(rezultatai.items()):
    reikšmės = [r[m] for m in metrikos]
    axes[1, 0].bar(x + j * plotis, reikšmės, plotis,
                   label=pav, color=spalvos[j], alpha=0.85)

axes[1, 0].set_title("Metrikų palyginimas")
axes[1, 0].set_xticks(x + plotis)
axes[1, 0].set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(axis="y", alpha=0.3)

# 5e: Treniravimo laikas
pavadinimai = list(rezultatai.keys())
laikai = [rezultatai[p]["laikas"] for p in pavadinimai]
axes[1, 1].barh(pavadinimai, laikai, color=spalvos, alpha=0.85)
axes[1, 1].set_title("Treniravimo laikas (s)")
axes[1, 1].set_xlabel("Sekundės")
axes[1, 1].grid(axis="x", alpha=0.3)

# 5f: Classification report geriausiam modeliui
axes[1, 2].axis("off")
ataskaita = classification_report(
    y_test, rezultatai[geriausias]["y_pred"],
    target_names=["Katė", "Šuo"]
)
axes[1, 2].text(0.05, 0.95, f"Geriausias modelis:\n{geriausias}\n\n{ataskaita}",
                transform=axes[1, 2].transAxes,
                fontsize=9, verticalalignment="top",
                fontfamily="monospace")
axes[1, 2].set_title("Classification report")

plt.tight_layout()
plt.savefig("modeliu_palyginimas.png", dpi=150, bbox_inches="tight")
print("\nGrafikai išsaugoti: modeliu_palyginimas.png")
print("\n✓ BAIGTA – rezultatus naudok ataskaitai")