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


SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
}


def create_tokenizer(
    corpus_path: str,
    save_dir: str,
    vocab_size: int = 65536,
):
    """创建针对特定语言优化的分词器"""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"语料文件 {corpus_path} 不存在")

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
        pre_tokenizers.Punctuation(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(
            add_prefix_space=True,  # 关键修改：处理空格前缀
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
        model_max_length=256,
        padding_side="right",
        truncation_side="right",
        do_lower_case=True
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "BPE_UniversalTokenizer")
    fast_tokenizer.save_pretrained(save_path)
    return fast_tokenizer


def load_tokenizer(path: str):
    """按语言加载分词器"""
    path = f"{path}/BPE_UniversalTokenizer"
    if not os.path.exists(path):
        raise FileNotFoundError("找不到分词器，请先训练")
    return PreTrainedTokenizerFast.from_pretrained(path)


if __name__ == "__main__":
    tokenizer = create_tokenizer(
        corpus_path="data/corpus.txt",
        save_dir="model/tokenizers",
        vocab_size=65536
    )

    tokenizer = load_tokenizer("model/tokenizers")

    # 测试中文处理
    zh_text = "到目前为止你认为你读过多少书？"
    zh_encoding = tokenizer(zh_text)
    print("中文编码:", zh_encoding.tokens())
    print("中文解码:", tokenizer.decode(zh_encoding.input_ids))

    # 测试英文处理
    en_text = "How many books do you think you've read so far?"
    en_encoding = tokenizer(en_text)
    print("\n英文编码:", en_encoding.tokens())
    print("英文解码:", tokenizer.decode(en_encoding.input_ids))
