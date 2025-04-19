import json

# convert json to pure text
with open("data/corpus.json", "r", encoding='utf8') as f:
    corpus = json.load(f)

with open("data/corpus.txt", "w", encoding='utf8') as f:
    for i in corpus:
        f.write(i["en"] + "\n")
        f.write(i["zh"] + "\n")
