# UVBF FAQ Chatbot (CLI prototype)

Ce mini-projet charge trois fichiers :
- `FAQ_UV-BF.csv`
- `ner_uvbf.json`
- `templates_FAQ_uvbf.json`

et fournit un chatbot en ligne de commande (pas de Streamlit).

## Installation
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer
Place ces 3 fichiers à la racine du projet (ou ajuste `config.yaml`), puis :
```bash
python -m src.app
```

## Notes
- Intentions: règles simples (keywords) pour démarrer.
- NER: regex issues de `ner_uvbf.json` (champ `patterns`). Si vide, pas de règle.
- Recherche: TF-IDF + similarité cosinus sur question/variantes/réponse.
- Génération: si `required_entities` manquent, le bot demande une précision.
