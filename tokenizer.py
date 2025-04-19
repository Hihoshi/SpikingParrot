from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast
import json


# from json to pure text
with open("corpus.json", "r", encoding='utf8') as f:
    corpus = json.load(f)

with open("corpus.txt", "w", encoding='utf8') as f:
    for i in corpus["pairs"]:
        f.write(i["zh"]+"\n")
        f.write(i["en"]+"\n")

# train new tokenizer from current database
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.NFD(),
        normalizers.StripAccents()
    ]
)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel()
    ]
)
tokenizer.post_processor = processors.Sequence(
    [
        processors.ByteLevel(),
        processors.TemplateProcessing(
            single="$0 [EOS]",
            pair="$A [EOS] $B:1 [EOS]:1",
            special_tokens=[("[EOS]", 1)],
        )
    ]
)
tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, use_regex=True)
special_tokens = ["[EOS]", "[PAD]"]
trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=24576)
tokenizer.train(["corpus.txt"], trainer=trainer)
tokenizer.save("BPE_tokenizer.json")


# load pretrained tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer = tokenizer.from_file("BPE_tokenizer.json")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    eos_token="[EOS]",
    pad_token="[PAD]"
)


if __name__ == "__main__":
    encoding = tokenizer("昨日春风压东风，旧日王侯换新容。")
    print(encoding)
    print(tokenizer.decode(encoding["input_ids"]))
