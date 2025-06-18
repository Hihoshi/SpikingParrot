# inference.py
import torch
import pandas as pd
from sacrebleu import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path
from MyTokenizer import load_tokenizer


def load_dataset(path):
    df = pd.read_parquet(path)
    try:
        return list(zip(
            df['translation'].apply(lambda x: x['en']),
            df['translation'].apply(lambda x: x['zh'])
        ))
    except KeyError as e:
        raise ValueError(f"Missing required translation key: {e}")


def make_batches(items, batch_size):
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


class BLEUEvaluator:
    def __init__(self, model, en_tokenizer, zh_tokenizer, device, batch_size, src_max_length, tgt_max_length):
        self.model = model
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.device = device
        self.batch_size = batch_size
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.pad_token_id = en_tokenizer.pad_token_id
        self.bos_token_id = zh_tokenizer.bos_token_id
        self.eos_token_id = zh_tokenizer.eos_token_id

    def preprocess_batch(self, batch):
        valid_samples = []
        discarded = 0
        for en_text, zh_ref in batch:
            try:
                if len(self.en_tokenizer.encode(en_text)) <= self.src_max_length:
                    valid_samples.append((en_text, zh_ref))
                else:
                    discarded += 1
            except Exception as e:
                print(f"Tokenization error: {e}")
                discarded += 1
        return valid_samples, discarded

    def encode_batch(self, en_batch):
        batch_encoded = [self.en_tokenizer.encode(text) for text in en_batch]
        src_ids = [torch.tensor(x, device=self.device) for x in batch_encoded]
        return pad_sequence(
            src_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )

    def decode_predictions(self, outputs):
        return [
            self.zh_tokenizer.decode(output, skip_special_tokens=True).replace(" ", "")
            for output in outputs
        ]

    def write_comparison(self, output_path, en_batch, zh_batch, hypos_batch):
        with open(output_path, 'a', encoding='utf-8') as f:
            for en, zh, hyp in zip(en_batch, zh_batch, hypos_batch):
                f.write(f"EN: {en}\nZH_REF: {zh}\nZH_HYP: {hyp}\n\n")

    def evaluate(self, dataset_path, output_path="translations_comparison.txt"):
        Path(output_path).unlink(missing_ok=True)
        
        data = load_dataset(dataset_path)
        hypotheses = []
        references = []
        total_discarded = 0

        batches = make_batches(data, self.batch_size)
        progress_bar = tqdm(
            batches, desc="Evaluating", unit="batch", colour="blue",
            bar_format='{l_bar}{bar:32}{r_bar}',
            dynamic_ncols=True,
        )

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
                    max_length=self.tgt_max_length
                )
                # outputs = self.model.beam_decode(
                #     src_ids,
                #     bos_token_id=self.bos_token_id,
                #     eos_token_id=self.eos_token_id,
                #     max_length=self.tgt_max_length,
                #     beam_size=8,
                #     length_penalty=0.95,
                #     repetition_penalty=1.05
                # )

            current_hypos = self.decode_predictions(outputs)
            hypotheses.extend(current_hypos)
            references.extend([[ref] for ref in zh_batch])
            
            self.write_comparison(output_path, en_batch, zh_batch, current_hypos)

            progress_bar.set_postfix({
                "Processed": f"{len(hypotheses)}/{len(data)}",
                "Discarded": total_discarded
            })

        bleu_score = corpus_bleu(hypotheses, references, lowercase=False, smooth_method="floor", tokenize="zh")
        print(
            f"\nEvaluation Complete\n"
            f"Total samples: {len(data)}\n"
            f"Valid samples: {len(hypotheses)}\n"
            f"Discarded samples: {total_discarded}\n"
            f"BLEU score: {bleu_score.score:.2f}\n"
            f"Comparison file saved at: {output_path}"
        )
        return bleu_score


def main():
    en_tokenizer = load_tokenizer("model/tokenizers/en")
    zh_tokenizer = load_tokenizer("model/tokenizers/zh")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = torch.load("model/checkpoints/latest_model.pt", map_location=device, weights_only=False)
    # model = torch.load("model/checkpoints/model_009.pt", map_location=device, weights_only=False)
    model = torch.load("temp/latest_model.pt", map_location=device, weights_only=False)
    model.eval().to(device)
    
    evaluator = BLEUEvaluator(
        model=model,
        en_tokenizer=en_tokenizer,
        zh_tokenizer=zh_tokenizer,
        device=device,
        batch_size=512,
        src_max_length=128,
        tgt_max_length=256
    )
    
    _ = evaluator.evaluate(
        "data/valid/validation-00000-of-00001.parquet",
        output_path="results/translation_comparison.txt"
    )


if __name__ == "__main__":
    main()
