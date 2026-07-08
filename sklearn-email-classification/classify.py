import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib

# Sklearn bibliotekos importas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Vizualizacija
import matplotlib.pyplot as plt
import seaborn as sns

# Užtikriname, kad katalogai egzistuoja
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
result_dir = Path("results")
result_dir.mkdir(exist_ok=True)

# 1. Duomenų įkėlimas
print("Įkeliamas duomenų rinkinys...")
df = pd.read_csv("spam_ham_dataset.csv")
print(f"Duomenų rinkinio struktūra: {df.shape}")

# 2. Duomenų paruošimas
print("Duomenys paruošiami...")
# Pašaliname nenaudojamus stulpelius
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Patikriname ir pašaliname trūkstamas reikšmes
initial_len = len(df)
df = df.dropna(subset=["text", "label_num"])
removed = initial_len - len(df)
if removed > 0:
    print(f"Pašalinta {removed} eilučių su trūkstamomis reikšmėmis.")
else:
    print("Trūkstamų reikšmių nerasta.")

# Padalijame į požymius (X) ir taikinį (y)
X = df["text"]
y = df["label_num"]

# Padalijame duomenis į mokymo ir testavimo rinkinius (75% mokymui, 25% testavimui)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=89, stratify=y
)
print(f"Mokymo rinkinio dydis: {X_train.shape[0]}, Testavimo rinkinio dydis: {X_test.shape[0]}")

# 3. Požymių išgavimas (TF-IDF)
print("Tekstas vektorizuojamas naudojant TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Modelio apmokymas (Logistinė regresija)
print("Apmokomas logistinės regresijos modelis...")
model = LogisticRegression(random_state=89, max_iter=1000)
start_time = time.time()
model.fit(X_train_vec, y_train)
end_time = time.time()
training_time = end_time - start_time
print(f"Modelis apmokytas per {training_time:.4f} sekundžių.")

# 5. Modelio įvertinimas
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
acc_percentage = acc * 100
report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
cnf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- REZULTATAI ---")
print(f"Tikslumas (Accuracy): {acc_percentage:.2f}%")
print("Klasifikavimo ataskaita (Classification Report):\n", report)
print("Painiavos matrica (Confusion Matrix):\n", cnf_matrix)

# 6. Painiavos matricos vizualizacija
class_names = ["Ham", "Spam"]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.xaxis.set_label_position("top")
ax.set_title("Painiavos matricos vizualizacija", y=1.1)
ax.set_ylabel("Tikroji klasė")
ax.set_xlabel("Prognozuota klasė")
plt.tight_layout()
plt.savefig(result_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Painiavos matricos grafikas išsaugotas: {result_dir / 'confusion_matrix.png'}")

# 7. Save results to a text file
tn, fp, fn, tp = cnf_matrix[0][0], cnf_matrix[0][1], cnf_matrix[1][0], cnf_matrix[1][1]

analysis = f"""EMAIL CLASSIFICATION RESULTS (SPAM / HAM)

Model: Logistic Regression
Features: TF-IDF (English stop words removed, max 5000 features)
Train samples: {X_train.shape[0]}
Test samples: {X_test.shape[0]}
Training time: {training_time:.4f} sec

Accuracy: {acc_percentage:.2f}%

Classification report:
{report}

Confusion matrix:
{cnf_matrix}
True negatives (Ham correct): {tn}
False positives (Ham marked as Spam): {fp}
False negatives (Spam marked as Ham): {fn}
True positives (Spam correct): {tp}

Analysis:
The model reached {acc_percentage:.2f}% accuracy on the test set.
Only {fp} ham email(s) were wrongly marked as spam, and only {fn} spam email(s) were missed.
Low false positives matter most here, since a real email getting blocked is worse than a spam email getting through.
"""

with open(result_dir / "results.txt", "w") as f:
    f.write(analysis)
print(f"Saved results to {result_dir / 'results.txt'}")

# 8. Save model and vectorizer
joblib.dump(model, models_dir / "model.pkl")
joblib.dump(vectorizer, models_dir / "vectorizer.pkl")
print("Model and vectorizer saved.")