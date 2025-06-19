import pandas as pd
import joblib
import re

# Chargement initial
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

# Découpe en phrases
def decoupe_en_phrases(texte):
    return [p for p in re.split(r'[.!?]\s*', texte.strip()) if p]

# Correction orthographique optionnelle (recommandée)
from spellchecker import SpellChecker
spell = SpellChecker(language='fr')

def corriger_texte(phrase):
    mots = phrase.split()
    corriges = [spell.correction(mot) or mot for mot in mots]
    return ' '.join(corriges)

# Boucle interactive
while True:
    texte = input("\nTexte à analyser ('q' pour quitter) :\n> ")
    if texte.lower() == 'q':
        break

    phrases = decoupe_en_phrases(texte)
    if not phrases:
        print("Texte vide.")
        continue

    ia_detectees = 0
    historique = []

    for phrase in phrases:
        phrase_corrigee = corriger_texte(phrase)
        vect = vectorizer.transform([phrase_corrigee])
        pred = model.predict(vect)[0]
        print(f"\nPhrase : {phrase}")
        print(f"Prédiction : {'IA' if pred == 1 else 'Humain'}")

        feedback = input("Est-ce correct ? (y/n) : ").lower().strip()
        label_final = pred
        if feedback == "n":
            vrai_label = input("Bon label ? (1=IA, 0=Humain) : ").strip()
            if vrai_label in ["0", "1"]:
                label_final = int(vrai_label)
            else:
                print("Label invalide, résultat initial conservé.")
        if label_final == 1:
            ia_detectees += 1

        # Sauvegarde automatique
        historique.append({"phrase": phrase_corrigee, "label": label_final})

    # Écriture auto. dans historique.csv
    df_historique = pd.DataFrame(historique)
    df_historique.to_csv("historique.csv", mode='a', index=False, header=False)

    pourcentage = (ia_detectees / len(phrases)) * 100
    print(f"\nCe texte contient environ {pourcentage:.2f}% de contenu IA détecté.")
