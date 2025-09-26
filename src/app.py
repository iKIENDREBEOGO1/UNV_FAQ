import yaml
from .loader import load_faq, load_json
from .ner import RegexNER
from .retriever import Retriever
from .intent import classify_intent
from .templates import TemplateManager

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    faq = load_faq(cfg["data"]["faq_csv"])
    ner_schema = load_json(cfg["data"]["ner_json"])
    templates = load_json(cfg["data"]["templates_json"])

    ner = RegexNER(ner_schema)
    retr = Retriever(faq["index_text"], **cfg.get("retriever", {}))
    tm = TemplateManager(templates)

    print("UVBF FAQ Chatbot — CLI")
    print("Tapez votre question (ou 'quit' pour sortir)")
    while True:
        try:
            q = input("\nVous: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin.")
            break
        if q.lower() in {"quit", "exit"}:
            break

        ents = ner.extract(q)
        intent = classify_intent(q)

        rendered = tm.render(intent, ents)
        if not rendered["need_more_info"] and rendered["text"]:
            print(f"\nBot (template:{intent}): {rendered['text']}")
            continue
        elif rendered["need_more_info"]:
            print(f"\nBot: {rendered['text']}")

        hits = retr.search(q, top_k=cfg["retriever"]["top_k"])
        if not hits:
            print("\nBot: Désolé, je n'ai pas trouvé d'information pertinente.")
        else:
            best_idx, score = hits[0]
            row = faq.iloc[best_idx]
            print(f"\nBot (FAQ: {row['question_canonique']} | score={score:.3f})")
            print(row['reponse'])
            if isinstance(row.get("liens"), str) and row["liens"]:
                print("Lien:", row["liens"].split(";")[0])

if __name__ == "__main__":
    main()
