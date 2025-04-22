import json
import jieba
from tqdm import tqdm

# convert json to pure text
with open("data/corpus.json", "r", encoding='utf8') as f:
    corpus = json.load(f)


with open("data/corpus.en", "w", encoding='utf8') as f:
    corpus = tqdm(
        corpus,
        unit="sentence",
        colour="green",
        bar_format='{l_bar}{bar:32}{r_bar}',
        dynamic_ncols=True
    )
    for i in corpus:
        f.write(i["en"] + "\n")


with open("data/corpus.zh", "w", encoding='utf8') as f:
    print(jieba.lcut("这是一个结巴中文分词测试"))
    print("ZH jieba tokenizer test complete...")
    corpus = tqdm(
        corpus,
        unit="sentence",
        colour="green",
        bar_format='{l_bar}{bar:32}{r_bar}',
        dynamic_ncols=True
    )
    for i in corpus:
        # # jieba pretokenize
        f.write(" ".join(jieba.lcut(i["zh"])) + "\n")
