from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
    processors
)
from transformers import PreTrainedTokenizerFast
import os
import jieba

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
}


def create_tokenizer(
    corpus_path: str,
    save_dir: str,
    language: str,
    vocab_size: int,
):
    """create a univeral bpe tokenizer"""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus file {corpus_path} not exists")

    tokenizer = Tokenizer(models.BPE(
        unk_token=SPECIAL_TOKENS["unk_token"]
    ))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.StripAccents(),
        normalizers.NFD(),
        normalizers.Lowercase(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(
            add_prefix_space=True,
            use_regex=True
        )
    ])

    trainer = trainers.BpeTrainer(
        special_tokens=list(SPECIAL_TOKENS.values()),
        vocab_size=vocab_size,
        min_frequency=4,
        show_progress=True
    )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    # 添加后处理器以自动添加特殊标记
    bos_token = SPECIAL_TOKENS["bos_token"]
    eos_token = SPECIAL_TOKENS["eos_token"]
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        pair=f"{bos_token} $A {eos_token} {bos_token} $B {eos_token}",
        special_tokens=[
            (bos_token, tokenizer.token_to_id(bos_token)),
            (eos_token, tokenizer.token_to_id(eos_token)),
        ],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        **SPECIAL_TOKENS,
        padding_side="right",
        truncation_side="right",
        do_lower_case=True
    )

    os.makedirs(save_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(save_dir)
    return fast_tokenizer


def load_tokenizer(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError("can not find tokenizer, plz train first")
    return PreTrainedTokenizerFast.from_pretrained(path)


if __name__ == "__main__":
    en_tokenizer = create_tokenizer(
        corpus_path="data/txt/corpus.en",
        save_dir="model/tokenizers/en",
        language="en",
        vocab_size=16384,
    )
    zh_tokenizer = create_tokenizer(
        corpus_path="data/txt/corpus.zh",
        save_dir="model/tokenizers/zh",
        language="zh",
        vocab_size=16384,
    )

    en_tokenizer = load_tokenizer("model/tokenizers/en")
    zh_tokenizer = load_tokenizer("model/tokenizers/zh")

    # 测试英文处理
    en_text = "How many books do you think you've read so far?"
    en_encoding = en_tokenizer(en_text)
    print("en encoding:", en_encoding.tokens())
    print("en decoding:", en_tokenizer.decode(en_encoding.input_ids))

    # 测试中文处理
    zh_text = "到目前为止你认为你读过多少书？"
    jieba.initialize()
    zh_text = " ".join(jieba.lcut(zh_text))
    zh_encoding = zh_tokenizer(zh_text)
    print("zh encoding:", zh_encoding.tokens())
    print("zh decoding:", zh_tokenizer.decode(zh_encoding.input_ids))
