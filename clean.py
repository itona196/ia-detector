import csv

# Lire data.csv et filtrer les lignes valides (2 champs)
with open("data.csv", "r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    lignes_valides = [row for row in reader if len(row) == 2]

# Réécrire data.csv avec uniquement les lignes correctes, et des guillemets
with open("data.csv", "w", encoding="utf-8", newline='') as outfile:
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    writer.writerows(lignes_valides)

print(f"✅ data.csv nettoyé avec succès. {len(lignes_valides)} ligne(s) conservée(s).")
