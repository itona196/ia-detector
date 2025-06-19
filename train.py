import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
import time
import os

def entrainer_modele():
    if not os.path.exists("historique.csv"):
        print("Aucune donnée à entraîner.")
        return

    df = pd.read_csv("historique.csv", names=["phrase", "label"])

    # Convertir explicitement les labels en entiers
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Vérification supplémentaire
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if len(df["label"].unique()) < 2:
        print("Pas assez de classes différentes pour entraîner.")
        return

    X, y = df["phrase"], df["label"]

    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    model = SGDClassifier(loss='log_loss')
    model.partial_fit(X_vect, y, classes=[0, 1])

    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump(model, "model.joblib")
    print("Modèle entraîné et sauvegardé.")

if __name__ == "__main__":
    while True:
        entrainer_modele()
        time.sleep(300)
