import json
import pandas as pd

def load_faq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["id","categorie","question_canonique","variantes","reponse","liens","source","derniere_mise_a_jour"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    df["variantes"] = df["variantes"].fillna("")
    df["reponse"] = df["reponse"].fillna("")
    df["question_canonique"] = df["question_canonique"].fillna("")
    df["index_text"] = (df["question_canonique"].astype(str) + " " + df["variantes"].astype(str) + " " + df["reponse"].astype(str))
    return df

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
