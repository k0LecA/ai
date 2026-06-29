# ============================================================
# VIENOS NUOTRAUKOS PROGNOZĖ
# ============================================================
# Naudojimas:
#   python 3_prognozė.py nuotrauka.jpg
#   python 3_prognozė.py C:/Users/mano/Downloads/katinas.png
# ============================================================

import sys
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# NUSTATYMAI – pakeisk į geriausią savo modelį
# ─────────────────────────────────────────
IMG_SIZE = (64, 64)
KLASĖS   = {0: "Katė", 1: "Šuo"}

# ─────────────────────────────────────────
# 1. NUOTRAUKOS KELIAS IŠ KOMANDŲ EILUTĖS
# ─────────────────────────────────────────
if len(sys.argv) < 2:
    print("Naudojimas: python 3_prognozė.py <nuotraukos_kelias>")
    print("Pvz.:       python 3_prognozė.py katinas.jpg")
    sys.exit(1)

kelias = sys.argv[1]

# ─────────────────────────────────────────
# 2. NUOTRAUKOS PARUOŠIMAS (tas pats procesas kaip treniravime)
# ─────────────────────────────────────────
print(f"Įkeliama: {kelias}")
img = Image.open(kelias).convert("RGB")
originalas = img.copy()

img = img.resize(IMG_SIZE, Image.LANCZOS)
arr = np.array(img) / 255.0
X_nauja = arr.flatten().reshape(1, -1)  # forma: (1, 12288)

# ─────────────────────────────────────────
# 3. MODELIŲ PERKROVIMAS IR PROGNOZĖ
# ─────────────────────────────────────────
# Perkrauname duomenis ir treniruojame iš naujo
print("Kraunami treniravimo duomenys...")
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

X_visi = np.vstack([X_train, X_test])
y_visi = np.concatenate([y_train, y_test])

# Standartizavimas – svarbu naudoti tuos pačius parametrus
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_visi)
X_nauja_sc = scaler.transform(X_nauja)

# Treniruojame visus tris modelius
modeliai = {
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                SVC(kernel="rbf", probability=True, random_state=42),
}

print("\nPrognozės:")
print("-" * 35)

prognozės = {}
for pav, modelis in modeliai.items():
    modelis.fit(X_train, y_train)
    pred  = modelis.predict(X_nauja_sc)[0]
    prob  = modelis.predict_proba(X_nauja_sc)[0] if hasattr(modelis, "predict_proba") else None
    prognozės[pav] = (pred, prob)

    if prob is not None:
        print(f"  {pav:<22}: {KLASĖS[pred]}  "
              f"(katė {prob[0]*100:.1f}% / šuo {prob[1]*100:.1f}%)")
    else:
        print(f"  {pav:<22}: {KLASĖS[pred]}")

# Daugumos balsas
balsai = [p[0] for p in prognozės.values()]
galutinis = max(set(balsai), key=balsai.count)
print(f"\n  Galutinis sprendimas: {KLASĖS[galutinis].upper()}")

# ─────────────────────────────────────────
# 4. VIZUALIZACIJA
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"Prognozė: {KLASĖS[galutinis]}", fontsize=14, fontweight="bold")

# Originali nuotrauka
axes[0].imshow(originalas)
axes[0].set_title("Įkelta nuotrauka")
axes[0].axis("off")

# Tikimybių grafikas (Random Forest)
rf_prob = prognozės["Random Forest"][1]
if rf_prob is not None:
    axes[1].barh(["Katė", "Šuo"], [rf_prob[0]*100, rf_prob[1]*100],
                 color=["#4C9BE8", "#E8754C"], alpha=0.85)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Tikimybė (%)")
    axes[1].set_title("Random Forest tikimybės")
    axes[1].axvline(50, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(axis="x", alpha=0.3)
    for i, v in enumerate([rf_prob[0]*100, rf_prob[1]*100]):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center")

plt.tight_layout()
plt.savefig("prognozė_rezultatas.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nRezultatas išsaugotas: prognozė_rezultatas.png")