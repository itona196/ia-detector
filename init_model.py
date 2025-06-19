import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib

df = pd.read_csv("data.csv")
X, y = df["texte"], df["label"]

vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

model = SGDClassifier(loss='log_loss')
model.partial_fit(X_vect, y, classes=[0, 1])

joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(model, "model.joblib")

with open("historique.csv", "w") as f:
    f.write("phrase,label\n")

print("Réinitialisation terminée correctement.")
