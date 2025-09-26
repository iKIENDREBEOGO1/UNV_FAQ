import regex as re

class RegexNER:
    def __init__(self, ner_schema: dict):
        self.patterns = {}
        for ent in ner_schema.get("entities", []):
            name = ent.get("name")
            pat_list = ent.get("patterns", [])
            compiled = []
            for pat in pat_list:
                try:
                    compiled.append(re.compile(pat, re.IGNORECASE))
                except re.error:
                    pass
            if compiled:
                self.patterns[name] = compiled

    def extract(self, text: str):
        found = {}
        for name, pats in self.patterns.items():
            for p in pats:
                for m in p.finditer(text or ""):
                    val = m.group(0)
                    found.setdefault(name, [])
                    if val not in found[name]:
                        found[name].append(val.strip())
        return found
