

import os
from pathlib import Path
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import joblib

RND = 42
np.random.seed(RND)
sns.set(style="whitegrid", context="notebook")

# ---------- Paths ----------
BASE = Path(".")
DATA_DIR = BASE
FIG_DIR = BASE / "figures"
MODEL_DIR = BASE / "models"
OUT_DIR = BASE / "output"

for p in (FIG_DIR, MODEL_DIR, OUT_DIR):
    p.mkdir(exist_ok=True, parents=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# ---------- Helpers ----------
def load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No se encontro {p}. Pon el CSV en el directorio o ajusta la ruta.")
    return pd.read_csv(p)

# limpieza sencilla (sin dependencias externas)
def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # quitar urls, mentions, hashtags (solo simbolo), emojis aproximado
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#", " ", t)
    # eliminar puntuacion y dígitos sueltos (dejamos numeros importantes si quisiera)
    t = re.sub(rf"[{re.escape(string.punctuation)}]", " ", t)
    t = re.sub(r"\d+", " ", t)
    # normalizar espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str):
    return text.split()

def plot_save_class_dist(df: pd.DataFrame, out: Path):
    vc = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    pal = ["#3182bd", "#e6550d"]
    sns.barplot(x=vc.index.astype(str), y=vc.values, palette=pal[:len(vc)], ax=ax)
    total = vc.sum()
    for p in ax.patches:
        h = p.get_height()
        pct = h / total * 100
        ax.annotate(f"{int(h)}\n({pct:.1f}%)", (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=10)
    ax.set_title("Distribucion de clases (0=no desastre, 1=desastre)")
    ax.set_xlabel("target")
    ax.set_ylabel("Numero de tweets")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_save_length_hist(df: pd.DataFrame, out: Path):
    maxlen = min(280, int(df["text_char_len"].max()))
    bins = np.linspace(0, maxlen, 30)
    fig, ax = plt.subplots(figsize=(9,4))
    for cls, color in [(0, "#3182bd"), (1, "#e6550d")]:
        subset = df.loc[df["target"]==cls, "text_char_len"].clip(upper=maxlen)
        ax.hist(subset, bins=bins, alpha=0.35, label=f"class {cls}", color=color)
    ax.set_title("Longitud de tweets (caracteres) por clase")
    ax.set_xlabel("Longitud (caracteres)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def top_tokens(df: pd.DataFrame, cls:int, topn:int=25):
    c = Counter()
    for txt in df.loc[df["target"]==cls, "text_clean"]:
        c.update(tokenize(txt))
    return c.most_common(topn)

def plot_top_tokens(pairs, out: Path, title:str):
    if not pairs:
        return
    tokens = [p for p,_ in pairs][::-1]
    freqs = [f for _,f in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(tokens)*0.2)))
    ax.barh(tokens, freqs, color="#e6550d" if "Desastre" in title else "#3182bd")
    ax.set_xlabel("Frecuencia")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def save_confusion_matrix(cm, out: Path, title="CM"):
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

# ---------- Main ----------
def main():
    print("Cargando train.csv ...")
    df = load_csv(TRAIN_PATH)

    # asegurar columnas basicas
    if "text" not in df.columns or "target" not in df.columns:
        raise RuntimeError("train.csv debe contener columnas 'text' y 'target'")

    # EDA básico
    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].apply(simple_clean)
    df["text_char_len"] = df["text"].map(len)
    df["text_word_len"] = df["text_clean"].map(lambda s: len(s.split()))

    print("Guardando figuras EDA en ./figures ...")
    plot_save_class_dist(df, FIG_DIR / "01_class_distribution.png")
    plot_save_length_hist(df, FIG_DIR / "02_length_hist.png")

    # Top tokens
    top1 = top_tokens(df, 1, topn=25)
    top0 = top_tokens(df, 0, topn=25)
    plot_top_tokens(top1, FIG_DIR / "03_top_tokens_desastre.png", "Top tokens — Desastre (1)")
    plot_top_tokens(top0, FIG_DIR / "04_top_tokens_nodesastre.png", "Top tokens — No desastre (0)")

    # Simple modeling pipeline: TF-IDF (1-2) + three classifiers
    X = df["text_clean"]
    y = df["target"].astype(int)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RND)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RND, solver="liblinear"),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND),
        "nb": MultinomialNB()
    }

    results = []
    for name, clf in models.items():
        print(f"\nEntrenando {name} ...")
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        # guardar modelo
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

        # evaluar
        y_pred = pipe.predict(X_va)
        y_proba = pipe.predict_proba(X_va)[:, 1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_va, y_pred)
        prec = precision_score(y_va, y_pred, zero_division=0)
        rec = recall_score(y_va, y_pred, zero_division=0)
        f1 = f1_score(y_va, y_pred, zero_division=0)
        roc = roc_auc_score(y_va, y_proba) if y_proba is not None else np.nan

        print(f"Modelo {name} -> acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} roc_auc={roc if not np.isnan(roc) else 'NA'}")
        print(classification_report(y_va, y_pred, digits=3))

        # confusion matrix figura
        cm = confusion_matrix(y_va, y_pred)
        save_confusion_matrix(cm, FIG_DIR / f"cm_{name}.png", title=f"CM {name}")

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc
        })

    # guardar metrics.csv
    metrics_df = pd.DataFrame(results).set_index("model")
    metrics_df.to_csv(MODEL_DIR / "metrics.csv")
    print("\nMétricas guardadas en models/metrics.csv")

    # Si hay test.csv: generar submission usando el mejor modelo por f1
    if TEST_PATH.exists():
        print("\nGenerando submission con el mejor modelo (por F1)...")
        test = load_csv(TEST_PATH)
        best = metrics_df["f1"].idxmax()
        print("Mejor modelo:", best)
        model = joblib.load(MODEL_DIR / f"{best}.joblib")
        # preparar texto
        test_text = test["text"].astype(str).apply(simple_clean)
        preds = model.predict(test_text)
        sub = pd.DataFrame({"id": test["id"], "target": preds.astype(int)})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print("Submission guardada en output/submission.csv")
    else:
        print("\nNo se encontró test.csv — salto generación de submission.")

    print("\n¡Listo! Figuras y modelos guardados.")
    print(f"- Figuras en: {FIG_DIR}")
    print(f"- Modelos y metrics en: {MODEL_DIR}")
    print(f"- Submission (si aplica) en: {OUT_DIR}")

if __name__ == "__main__":
    main()
