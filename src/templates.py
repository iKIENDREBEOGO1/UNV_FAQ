class TemplateManager:
    def __init__(self, templates):
        self.templates = {item["intent"]: item for item in templates.get("intents", [])}

    def render(self, intent, entities):
        t = self.templates.get(intent)
        if not t:
            return {"text": "", "need_more_info": False, "missing": []}
        required = t.get("required_entities", [])
        missing = [e for e in required if e not in entities]
        if missing:
            prompt = t.get("fallback_prompt") or ("Précisez: " + ", ".join(missing))
            return {"text": prompt, "need_more_info": True, "missing": missing}

        values = {}
        values.update(t.get("defaults", {}))  # ✅ prérempli
        for k in ["NIVEAU","SEMESTRE","MONTANT","MODE_PAIEMENT","CONTACT","SERVICE","TYPE_EXAMEN"]:
            if k in entities and entities[k]:
                values[k] = entities[k][0]

        liens = t.get("default_links", [])
        values["LIEN"] = liens[0] if liens else ""

        text = t.get("template_text", "")
        out = text.format(**{k: values.get(k, "") for k in values})

        # ✅ suffixe conditionnel
        suffix = t.get("contact_suffix", "")
        if suffix and (values.get("SERVICE") or values.get("CONTACT")):
            out += suffix.format(**{k: values.get(k, "") for k in values})

        return {"text": out, "need_more_info": False, "missing": []}

