import regex as re

INTENT_RULES = [
    ("frais_inscription", [r"\bfrais\b|inscription|payer|paiement|versement"]),
    ("acces_plateforme", [r"plateforme|lms|moodle|mot de passe|connexion|connecter|login|se connecter"]),
    ("examens_modalites", [r"examen|évaluation|eval|en ligne|présentiel|sur table|modalit"]),
    ("info_generale_uvbf", [r"uv\-?bf|université virtuelle|formation|offre|filière|publique|privée"]),
]

def classify_intent(text: str) -> str:
    t = (text or "").lower()
    for intent, patterns in INTENT_RULES:
        for pat in patterns:
            if re.search(pat, t):
                return intent
    if re.search(r"examen|évaluation|sur table", t):
        return "examens_modalites"
    if re.search(r"frais|inscription|paiement|versement", t):
        return "frais_inscription"
    if re.search(r"mot de passe|plateforme|connexion|login", t):
        return "acces_plateforme"
    return "info_generale_uvbf"
