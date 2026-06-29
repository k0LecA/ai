# ============================================================
# NUOTRAUKŲ PARUOŠIMAS – Cats vs Dogs klasifikacija
# ============================================================
# Struktūra:
#   dataset/
#     cats/  *.jpg / *.png
#     dogs/  *.jpg / *.png
# ============================================================

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# NUSTATYMAI
# ─────────────────────────────────────────
DATA_DIR   = "animals"        # pagrindinis aplankas
IMG_SIZE   = (64, 64)         # resize iš 512x512 → 64x64
KLASĖS     = {"cats": 0, "dogs": 1}
RANDOM     = 42

# ─────────────────────────────────────────
# 1. NUOTRAUKŲ NUSKAITYMAS IR PARUOŠIMAS
# ─────────────────────────────────────────
X = []  # požymiai
y = []  # etiketės
klaidos = []

print("Skaitoma nuotraukos...")
for klasė_pav, etiketė in KLASĖS.items():
    aplankas = os.path.join(DATA_DIR, klasė_pav)
    failai = [f for f in os.listdir(aplankas)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"  {klasė_pav}: {len(failai)} nuotraukų")

    for failo_pav in failai:
        kelias = os.path.join(aplankas, failo_pav)
        try:
            img = Image.open(kelias)

            # RGB – pašalina alfa kanalą (PNG) ir pilkos spalvos nuotraukas
            img = img.convert("RGB")

            # Resize: 512x512 → 64x64 (greičiausia ir paprasčiausia)
            img = img.resize(IMG_SIZE, Image.LANCZOS)

            # Numpy masyvas: forma (64, 64, 3)
            arr = np.array(img)

            # Normalizavimas: pikseliai 0–255 → 0.0–1.0
            arr = arr / 255.0

            # Flatten: (64, 64, 3) → (12288,)  [64*64*3]
            X.append(arr.flatten())
            y.append(etiketė)

        except Exception as e:
            klaidos.append(failo_pav)

print(f"\nIš viso įkelta: {len(X)} nuotraukų")
if klaidos:
    print(f"Nepavyko įkelti: {len(klaidos)} failų: {klaidos[:5]}")

# ─────────────────────────────────────────
# 2. NUMPY MASYVAI
# ─────────────────────────────────────────
X = np.array(X)   # forma: (1000, 12288)
y = np.array(y)   # forma: (1000,)

print(f"\nX forma: {X.shape}  →  {X.shape[0]} nuotr. × {X.shape[1]} požymių")
print(f"y forma: {y.shape}")
print(f"Klasių pasiskirstymas: katės={sum(y==0)}, šunys={sum(y==1)}")

# ─────────────────────────────────────────
# 3. KLASIŲ PASISKIRSTYMO GRAFIKAS
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Duomenų analizė – Cats vs Dogs", fontweight="bold")

# Pasiskirstymas
axes[0].bar(["Katės (0)", "Šunys (1)"],
            [sum(y==0), sum(y==1)],
            color=["#4C9BE8", "#E8754C"], edgecolor="none")
axes[0].set_title("Klasių pasiskirstymas")
axes[0].set_ylabel("Kiekis")
axes[0].grid(axis="y", alpha=0.3)

# Pikselių intensyvumo pasiskirstymas
axes[1].hist(X[y==0].flatten(), bins=50, alpha=0.5,
             color="#4C9BE8", label="Katės", density=True)
axes[1].hist(X[y==1].flatten(), bins=50, alpha=0.5,
             color="#E8754C", label="Šunys", density=True)
axes[1].set_title("Pikselių intensyvumas")
axes[1].set_xlabel("Reikšmė (0.0 – 1.0)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_grafikai.png", dpi=150, bbox_inches="tight")
print("\nGrafikai išsaugoti: eda_grafikai.png")

# ─────────────────────────────────────────
# 4. TRAIN / TEST PADALIJIMAS
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM,
    stratify=y        # išlaiko klasių proporcijas
)

print(f"\nTrain: {X_train.shape[0]} nuotr.  |  Test: {X_test.shape[0]} nuotr.")
print(f"Train klasės: katės={sum(y_train==0)}, šunys={sum(y_train==1)}")
print(f"Test  klasės: katės={sum(y_test==0)},  šunys={sum(y_test==1)}")

# ─────────────────────────────────────────
# 5. STANDARTIZAVIMAS
# ─────────────────────────────────────────
# Svarbu: fit tik ant train, transform ant abiejų
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nPo standartizavimo:")
print(f"  Train vidurkis ≈ {X_train_sc.mean():.4f}  (turėtų būti ≈ 0)")
print(f"  Train std     ≈ {X_train_sc.std():.4f}  (turėtų būti ≈ 1)")

# ─────────────────────────────────────────
# 6. IŠSAUGOJIMAS
# ─────────────────────────────────────────
np.save("X_train.npy", X_train_sc)
np.save("X_test.npy",  X_test_sc)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)

print("\n✓ Išsaugota: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("\nKitas žingsnis: 2_modeliai.py  →  SVM + Random Forest treniravimas")