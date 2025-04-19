import json

with open("data/corpus.json", "r", encoding='utf8') as f:
    corpus = json.load(f)

filtered_corpus = [item for item in corpus if len(item['zh'].split()) <= 24 and len(item['en'].split()) <= 64]


with open("data/filtered_corpus.json", "w", encoding='utf8') as f:
    json.dump(filtered_corpus[int(len(filtered_corpus) / 8 * 7):], f, ensure_ascii=False, indent=2)
