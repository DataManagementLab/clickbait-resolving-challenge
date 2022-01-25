import os
import json
import nltk as nltk

ENTRY_SETS = ['train', 'dev', 'test']
DATA_PATH = "../data/"
RESULT_PATH = "../data/baseline_results/first_sentence/"

def setup():
    pass

def _get_first_sentence(text):
    if text != "":
        if text[0] == ".":
            text = text[1:].strip()
        sentences = nltk.tokenize.sent_tokenize(text, language='english')
        if len(sentences) > 0:
            return sentences[0]
    return "-"

def compute(entries):
    return [{"id": entry["id"], "answer": _get_first_sentence(entry["text"])} for entry in entries]


if __name__ == "__main__":
    os.makedirs(RESULT_PATH, exist_ok=True)

    for s in ENTRY_SETS:
        with open(f"{DATA_PATH}final_{s}.json", "r") as entry_file:
            results = compute(json.load(entry_file))

        with open(f"{RESULT_PATH}{s}.json", "w") as result_file:
            json.dump(results, result_file, indent=2, ensure_ascii=False)
