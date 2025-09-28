import regex as re

INTENT_RULES = [
    ("frais_inscription", [r"\bfrais\b", r"inscription", r"payer", r"paiement", r"versement"]),
    ("acces_plateforme", [r"plateforme", r"lms", r"moodle", r"mot de passe", r"connexion", r"connecter", r"login", r"se connecter"]),
    ("examens_modalites", [r"examen", r"évaluation", r"eval", r"en ligne", r"présentiel", r"sur table", r"modalit"]),
    ("info_generale_uvbf", [r"uv\-?bf", r"université virtuelle", r"formation", r"offre", r"filière", r"publique", r"privée"]),
]

def classify_intent(text: str) -> str:
    t = (text or "").lower()
    for intent, patterns in INTENT_RULES:
        for pat in patterns:
            if re.search(pat, t):
                return intent
    return "info_generale_uvbf"
