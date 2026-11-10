import re


class TokenizerBM25:
    def __init__(self):
        self.pattern = re.compile(r'\b\w+\b')

    def tokenize(self, text: str) -> list[str]:
        return self.pattern.findall(text.lower())
