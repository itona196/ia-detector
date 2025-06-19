# IA Detector

Ce projet est un détecteur de texte généré par intelligence artificielle. Il utilise un modèle de machine learning (régression logistique) entraîné sur des textes IA et humains.

## Structure du projet

```
ia-detector/
├── data.csv                # Données d'entraînement
├── model.joblib           # Modèle entraîné
├── vectorizer.joblib      # Vecteur TF-IDF sauvegardé
├── main.py                # Analyse d'un texte en entrée
├── train.py               # Entraînement du modèle
├── README.md              # Documentation du projet
```

## Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/itona196/ia-detector.git
cd ia-detector
```

2. Créer un environnement virtuel :

```bash
python -m venv venv
venv\Scripts\activate   # Sous Windows
# source venv/bin/activate  # Sous Linux/macOS
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

Si le fichier `requirements.txt` est manquant, installez manuellement :

```
scikit-learn
pandas
joblib
```

## Utilisation

1. Entraîner le modèle :

```bash
python train.py
```

2. Lancer l’analyse d’un texte :

```bash
python main.py
```

Vous pouvez ensuite saisir un texte pour savoir s’il a été écrit par une intelligence artificielle.

## Exemple de sortie

```
> Bonjour, comment vas-tu aujourd’hui ?
Le texte semble avoir été généré par une IA.
```

## Données

Le fichier `data.csv` contient deux colonnes :
- `texte` : le contenu textuel à analyser
- `label` : 0 pour humain, 1 pour IA
