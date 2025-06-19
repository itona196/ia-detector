from flask import Flask, request, render_template_string
from docx import Document
import io
from joblib import load
import os

# Charger le modèle et le vectorizer
model = load("model.joblib")
vectorizer = load("vectorizer.joblib")

app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <title>IA Detector</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f6f8;
            margin: 0;
            padding: 40px;
        }
        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 30px 40px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h2 {
            color: #333;
        }
        input[type="file"] {
            padding: 6px;
            margin-bottom: 10px;
        }
        input[type="submit"] {
            background-color: #007BFF;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        pre {
            background-color: #f7f7f7;
            padding: 15px;
            border: 1px solid #ccc;
            white-space: pre-wrap;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            background-color: #e2f0cb;
            border-left: 5px solid #7bb661;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Uploader un fichier texte ou Word (.txt ou .docx)</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="fichier" accept=".txt,.docx" required>
            <br>
            <input type="submit" value="Analyser">
        </form>
        {% if texte %}
            <div class="result">
                <strong>Résultat de l'analyse IA :</strong><br>
                Ce texte contient environ <strong>{{ pourcentage }}%</strong> de contenu généré par une IA.
            </div>
            <h3>Contenu du fichier :</h3>
            <pre>{{ texte }}</pre>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def upload_file():
    texte = None
    pourcentage = None
    if request.method == "POST":
        fichier = request.files.get("fichier")
        if fichier:
            if fichier.filename.endswith(".txt"):
                texte = fichier.read().decode("utf-8")
            elif fichier.filename.endswith(".docx"):
                doc = Document(io.BytesIO(fichier.read()))
                texte = "\n".join([p.text for p in doc.paragraphs])
            if texte:
                vect = vectorizer.transform([texte])
                proba = model.predict_proba(vect)[0][1]  # proba d'être IA
                pourcentage = round(proba * 100, 2)
    return render_template_string(HTML_TEMPLATE, texte=texte, pourcentage=pourcentage)

if __name__ == "__main__":
    app.run(debug=True)
