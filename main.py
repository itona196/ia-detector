import pandas as pd
import joblib
import re

# Chargement du modèle et du vectorizer
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

# Fonction pour découper le texte en phrases
def decoupe_en_phrases(texte):
    return [p for p in re.split(r'[.!?]\s*', texte.strip()) if p]

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
    nouvelles_donnees = []

    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue

        vect = vectorizer.transform([phrase])
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
        elif feedback not in ["y", ""]:
            print("Réponse invalide. La phrase ne sera pas enregistrée.")
            continue

        if label_final == 1:
            ia_detectees += 1

        nouvelles_donnees.append({"texte": phrase, "label": label_final})

    if nouvelles_donnees:
        df = pd.DataFrame(nouvelles_donnees)
        df.to_csv(
            "data.csv",
            mode='a',
            index=False,
            header=False,
            quoting=1,         # csv.QUOTE_ALL
            quotechar='"',
            lineterminator='\n'
        )

        pourcentage = (ia_detectees / len(nouvelles_donnees)) * 100
        print(f"\nCe texte contient environ {pourcentage:.2f}% de contenu IA détecté.")
    else:
        print("Aucune phrase enregistrée.")
