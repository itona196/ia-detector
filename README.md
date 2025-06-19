# IA-Detector

Détection automatique de texte généré par intelligence artificielle à l'aide de Machine Learning (`scikit-learn`).  
Ce projet permet d'entraîner un modèle sur des textes humains/IA et d'analyser un texte pour détecter s’il contient de l’IA (en % ou en étiquette).

---

## Structure du projet

```
ia-detector/
│
├── main.py               # Interface utilisateur (analyse de texte)
├── train.py              # Entraînement du modèle
├── data.csv              # Données d'entraînement (texte, label)
├── model.joblib          # Modèle ML entraîné (généré après train)
├── vectorizer.joblib     # TF-IDF vectorizer (généré après train)
├── requirements.txt      # Dépendances Python
├── README.md             # Ce fichier
└── venv/                 # Environnement virtuel (non versionné)
```

---

## ⚙️ Installation

1. Cloner le projet :
```bash
git clone https://github.com/itona196/ia-detector.git
cd ia-detector
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
```

3. Activer l'environnement virtuel :

- **Windows (PowerShell)** :
```bash
.env\Scripts\Activate.ps1
```

- **Windows (cmd.exe)** :
```bash
venv\Scriptsctivate.bat
```

- **Linux/macOS** :
```bash
source venv/bin/activate
```

4. Installer les dépendances :
```bash
pip install -r requirements.txt
```

---

## Entraîner le modèle

Avant d’utiliser le détecteur, entraîner le modèle avec les données :

```bash
python train.py
```

Cela génère :

- `model.joblib`
- `vectorizer.joblib`

---

## Utilisation

Lancer le script principal pour analyser un texte :

```bash
python main.py
```

Tu pourras entrer du texte et obtenir une réponse comme :

```
Ce texte semble généré par une **IA**.
Ce texte semble écrit par un **humain**.
Ce texte contient environ **XX%** de contenu généré par IA.
```

---

## Format du fichier `data.csv`

```csv
texte,label
"Voici un exemple de texte écrit par un humain.",0
"Cet article a été généré automatiquement par une IA.",1
```

- `label` : `0` = humain, `1` = IA

---

## Dépendances clés

- scikit-learn
- joblib
- pandas
- pyspellchecker

---

## Objectifs futurs

- Détection du **pourcentage** de contenu IA dans un texte
- Apprentissage par **renforcement** ou **incrémental**
- Interface web (Flask ou Next.js)

---

## Dépannage

- `ModuleNotFoundError` → Activer l’environnement virtuel puis `pip install -r requirements.txt`
- `vectorizer.joblib` introuvable → Exécuter `python train.py` avant `main.py`
