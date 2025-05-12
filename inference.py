# inference.py
import torch
import pandas as pd
from sacrebleu import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path
from mytokenizer import load_tokenizer


def load_dataset(path):
    """加载数据集并返回(en_text, zh_text)元组列表"""
    df = pd.read_parquet(path)
    try:
        return list(zip(
            df['translation'].apply(lambda x: x['en']),
            df['translation'].apply(lambda x: x['zh'])
        ))
    except KeyError as e:
        raise ValueError(f"Missing required translation key: {e}")


def make_batches(items, batch_size):
    """创建批次时保持原文与译文的对应关系"""
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


class BLEUEvaluator:
    def __init__(self, model, en_tokenizer, zh_tokenizer, device, batch_size=256, max_length=48):
        self.model = model
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pad_token_id = en_tokenizer.pad_token_id
        self.bos_token_id = zh_tokenizer.bos_token_id
        self.eos_token_id = zh_tokenizer.eos_token_id

    def preprocess_batch(self, batch):
        """预处理并过滤批次数据"""
        valid_samples = []
        discarded = 0
        for en_text, zh_ref in batch:
            try:
                if len(self.en_tokenizer.encode(en_text)) <= self.max_length:
                    valid_samples.append((en_text, zh_ref))
                else:
                    discarded += 1
            except Exception as e:
                print(f"Tokenization error: {e}")
                discarded += 1
        return valid_samples, discarded

    def encode_batch(self, en_batch):
        """编码并填充批次数据"""
        batch_encoded = [self.en_tokenizer.encode(text) for text in en_batch]
        src_ids = [torch.tensor(x, device=self.device) for x in batch_encoded]
        return pad_sequence(
            src_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )

    def decode_predictions(self, outputs):
        """解码模型输出并格式化"""
        return [
            self.zh_tokenizer.decode(output, skip_special_tokens=True).replace(" ", "")
            for output in outputs
        ]

    def write_comparison(self, output_path, en_batch, zh_batch, hypos_batch):
        """写入对比结果到文件"""
        with open(output_path, 'a', encoding='utf-8') as f:
            for en, zh, hyp in zip(en_batch, zh_batch, hypos_batch):
                f.write(f"EN: {en}\nZH_REF: {zh}\nZH_HYP: {hyp}\n\n")

    def evaluate(self, dataset_path, output_path="translations_comparison.txt"):
        """执行完整的BLEU评估流程"""
        Path(output_path).unlink(missing_ok=True)  # 清空旧文件
        
        data = load_dataset(dataset_path)
        hypotheses = []
        references = []
        total_discarded = 0

        batches = make_batches(data, self.batch_size)
        progress_bar = tqdm(batches, desc="Evaluating", unit="batch")

        for batch in progress_bar:
            valid_samples, discarded = self.preprocess_batch(batch)
            total_discarded += discarded
            if not valid_samples:
                continue

            en_batch, zh_batch = zip(*valid_samples)
            src_ids = self.encode_batch(en_batch)

            with torch.no_grad():
                outputs = self.model.greedy_decode(
                    src_ids,
                    bos_token_id=self.bos_token_id,
                    eos_token_id=self.eos_token_id,
                    max_length=self.max_length + 2
                )

            current_hypos = self.decode_predictions(outputs)
            hypotheses.extend(current_hypos)
            references.extend([[ref] for ref in zh_batch])
            
            # 写入当前batch的对比结果
            self.write_comparison(output_path, en_batch, zh_batch, current_hypos)

            progress_bar.set_postfix({
                "Processed": f"{len(hypotheses)}/{len(data)}",
                "Discarded": total_discarded
            })

        bleu_score = corpus_bleu(hypotheses, references, lowercase=True)
        print(
            f"\nEvaluation Complete\n"
            f"Total samples: {len(data)}\n"
            f"Valid samples: {len(hypotheses)}\n"
            f"Discarded samples: {total_discarded}\n"
            f"BLEU score: {bleu_score.score:.2f}\n"
            f"对比文件已保存至: {output_path}"
        )
        return bleu_score


def main():
    en_tokenizer = load_tokenizer("model/tokenizers/en")
    zh_tokenizer = load_tokenizer("model/tokenizers/zh")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load("model/checkpoints/model_001.pt", map_location=device)
    model.eval().to(device)
    
    evaluator = BLEUEvaluator(
        model=model,
        en_tokenizer=en_tokenizer,
        zh_tokenizer=zh_tokenizer,
        device=device,
        batch_size=128,
        max_length=64
    )
    
    # 指定输出文件路径
    bleu_score = evaluator.evaluate(
        "data/corpus/parquet/valid/validation-00000-of-00001.parquet",
        output_path="results/translation_comparison.txt"
    )
    print(f"Final BLEU score: {bleu_score.score:.2f}")


if __name__ == "__main__":
    main()
