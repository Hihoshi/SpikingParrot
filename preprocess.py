# preprocess.py
import glob
import pandas as pd
import os
from typing import List, Dict
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from MyTokenizer import load_tokenizer
import logging
import jieba
from BucketManager import BucketManager
from MyDataset import MyDataset
from functools import partial


def _init_pool_worker(en_dir: str, zh_dir: str):
    global _global_en_tokenizer, _global_zh_tokenizer

    jieba_logger = logging.getLogger('jieba')
    jieba_logger.setLevel(logging.WARNING)
    
    _global_en_tokenizer = load_tokenizer(en_dir)
    _global_zh_tokenizer = load_tokenizer(zh_dir)

    jieba.initialize()


def _tokenize_item(item: Dict[str, str], max_length: int):
    try:
        global _global_en_tokenizer, _global_zh_tokenizer
        src = item["en"]
        src_ids = _global_en_tokenizer(
            src,
            truncation=False,
            add_special_tokens=True
        ).input_ids
        
        zh_words = jieba.lcut(item["zh"])
        tgt = " ".join(zh_words)
        tgt_ids = _global_zh_tokenizer(
            tgt,
            truncation=False,
            add_special_tokens=True
        ).input_ids
        
        if len(src_ids) > max_length or len(tgt_ids) > max_length:
            return None

        return {
            "src": np.array(src_ids, dtype=np.int16),
            "tgt": np.array(tgt_ids[:-1], dtype=np.int16),
            "label": np.array(tgt_ids[1:], dtype=np.int16),
        }
    except Exception as e:
        print(f"Error processing item: {e}", flush=True)
        return None


def tokenize(
    raw_data: List[Dict[str, str]],
    en_tokenizer_dir: str,
    zh_tokenizer_dir: str,
    max_length: int,
    num_workers: int
) -> List[Dict[str, np.ndarray]]:
    with Pool(
        processes=num_workers,
        initializer=_init_pool_worker,
        initargs=(en_tokenizer_dir, zh_tokenizer_dir)
    ) as pool:
        results = []
        for result in tqdm(
            pool.imap(partial(_tokenize_item, max_length=max_length), raw_data),
            total=len(raw_data),
            desc="Tokenizing",
            unit="pairs",
            colour='green',
            bar_format='{l_bar}{bar:32}{r_bar}'
        ):
            if result is not None:
                results.append(result)
        return results


def shuffle_data(input_dir: str):
    files = glob.glob(os.path.join(input_dir, "*.parquet"))
    print("Reading and merging parquet files...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    if not all(isinstance(row, dict) and {'en', 'zh'}.issubset(row) for row in df['translation']):
        raise ValueError("Invalid translation format")
    print("Shuffling merged data...")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    num_workers = 20

    train_df = shuffle_data("data/train")
    train_raw_data = [{'en': pair['en'], 'zh': pair['zh']} for pair in train_df['translation'].tolist()]
    train_tokenized = tokenize(
        train_raw_data,
        "model/tokenizers/en",
        "model/tokenizers/zh",
        num_workers=num_workers,
        max_length=128
    )

    bucket = BucketManager(
        cache_path="data/cache/train",
        processed_data=train_tokenized,
        num_cuts=20,
        min_samples=32768,
        max_samples=262144,
        force_rebuild=True
    )

    bucket.print_stats()

    valid_df = shuffle_data("data/valid")
    valid_raw_data = [{'en': pair['en'], 'zh': pair['zh']} for pair in valid_df['translation'].tolist()]
    valid_tokenized = tokenize(
        valid_raw_data,
        "model/tokenizers/en",
        "model/tokenizers/zh",
        num_workers=num_workers,
        max_length=128
    )

    valid_dataset = MyDataset(
        cache_path="data/cache/valid",
        processed_data=valid_tokenized,
        src_range=(0, 128),
        tgt_range=(0, 128),
        shard_idx=0
    )
