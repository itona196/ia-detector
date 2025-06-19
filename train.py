import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
import os

def entrainer_sur_data():
    if not os.path.exists("data.csv"):
        print("Fichier data.csv introuvable.")
        return

    # Lecture robuste avec guillemets et 2 colonnes
    try:
        df = pd.read_csv("data.csv", names=["texte", "label"], quotechar='"', engine="python")
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return

    # Nettoyage des labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Vérification des classes
    if len(df["label"].unique()) < 2:
        print("❌ Pas assez de classes (besoin à la fois de 0 et 1).")
        return

    # Séparation des données
    X, y = df["texte"], df["label"]

    # Vectorisation et entraînement
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    model = SGDClassifier(loss="log_loss")
    model.fit(X_vect, y)

    # Sauvegarde des modèles
    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump(model, "model.joblib")

    print("Modèle entraîné à partir de data.csv et sauvegardé avec succès.")

if __name__ == "__main__":
    entrainer_sur_data()
