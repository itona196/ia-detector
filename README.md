# IA Detector

Ce projet open-source permet de **détecter automatiquement si un texte a été rédigé par une intelligence artificielle ou un humain**. Il utilise un modèle de machine learning (régression logistique) entraîné à partir d’un jeu de données contenant des textes générés par des IA type IA et d'autres type humain.

## Structure du projet

```
ia-detector/
├── data.csv                        # Données d'entraînement (texte + label)
├── model.joblib                    # Modèle ML entraîné
├── vectorizer.joblib               # Vecteur TF-IDF sauvegardé
├── main.py                         # Script pour analyser un texte
├── train.py                        # Script d'entraînement du modèle
├── app_flask_with_ia_analysis.py   # Interface Flask (expérimentale)
├── requirements.txt                # Dépendances Python
└── README.md                       # Documentation du projet
```

## Installation

1. **Cloner le dépôt :**
```bash
git clone https://github.com/itona196/ia-detector.git
cd ia-detector
```

2. **Créer un environnement virtuel :**
```bash
python -m venv venv
venv\Scripts\activate   # Sous Windows
# source venv/bin/activate  # Sous macOS / Linux
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt

## Utilisation

### 1. Entraîner le modèle :
```bash
python train.py
```

### 2. Lancer l’analyse d’un texte (mode terminal) :
```bash
python main.py
```
Vous pourrez alors entrer un texte, et le programme vous indiquera s'il a probablement été écrit par une IA.

### Exemple :
```
> Bonjour, comment vas-tu aujourd’hui ?
Ce texte semble avoir été généré par une IA.
```

## Données

Le fichier `data.csv` contient deux colonnes :
- `texte` : le contenu textuel
- `label` : 0 = humain, 1 = IA

## Interface Web (optionnel)

Une version expérimentale avec Flask est disponible :
```bash
python app_flask_with_ia_analysis.py
```

## Limitations

- Modèle simple basé sur TF-IDF + régression logistique.
- Données à enrichir pour améliorer la précision.
- La détection repose uniquement sur des statistiques de vocabulaire, pas sur une compréhension du sens.

## Pistes d'amélioration

- Intégration d'un modèle plus complexe (BERT, GPT, etc.)
- Interface utilisateur web plus avancée
- Entraînement avec davantage de données issues de contextes variés (scolaire, forums, réseaux sociaux)
