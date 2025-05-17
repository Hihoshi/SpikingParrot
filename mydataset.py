import pandas as pd
import os
import hashlib
from tqdm import tqdm
import jieba
import pickle
import zstandard as zstd
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from multiprocessing import Pool
import glob
import gc


class MyDataset(Dataset):
    def __init__(
        self, en_tokenizer_dir: str, zh_tokenizer_dir: str,
        parquet_file: str, cache_dir: str,
        src_max_length: int, tgt_max_length: int,
        num_workers: int,
        zstd_level: int = 9
    ):
        self.en_tokenizer_dir = en_tokenizer_dir
        self.zh_tokenizer_dir = zh_tokenizer_dir
        self.parquet_file = parquet_file
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.num_workers = num_workers
        self.zstd_level = zstd_level

        file_hash = hashlib.md5(os.path.basename(parquet_file).encode()).hexdigest()[:8]
        cache_name = f"cached_{file_hash}_src{src_max_length}_tgt{tgt_max_length}.zst"
        self.cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    self.processed_data = pickle.load(reader)
        else:
            if not os.path.exists(parquet_file):
                raise ValueError(f"Parquet file not found: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            raw_data = [{'en': pair['en'], 'zh': pair['zh']} for pair in df['translation'].tolist()]
            
            file_name = os.path.basename(parquet_file)
            jieba.initialize()
            print(f"Creating cache into {parquet_file}...")
            self.processed_data = self._parallel_process(raw_data, file_name)
            
            del df, raw_data
            gc.collect()
            
            print("Saving cache, please wait...")
            cctx = zstd.ZstdCompressor(level=self.zstd_level)
            with open(self.cache_path, 'wb') as f:
                with cctx.stream_writer(f) as compressor:
                    pickle.dump(
                        self.processed_data,
                        compressor,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
            print(f"Cache saved at: {self.cache_path}")

    def _parallel_process(self, raw_data, file):
        num_processes = self.num_workers
        with Pool(
            processes=num_processes,
            initializer=self._init_pool,
            initargs=(
                self.en_tokenizer_dir,
                self.zh_tokenizer_dir,
                self.src_max_length,
                self.tgt_max_length
            )
        ) as pool:
            results = []
            for result in tqdm(
                pool.imap(self._process_item, raw_data),
                total=len(raw_data),
                desc=f"Tokenizing {file}",
                unit="pairs",
                colour='green',
                bar_format='{l_bar}{bar:32}{r_bar}',
                dynamic_ncols=True,
                leave=False,
            ):
                if result is not None:
                    results.append(result)
            return results

    @staticmethod
    def _init_pool(en_dir, zh_dir, src_max_len, tgt_max_len):
        global en_tokenizer, zh_tokenizer, src_max_length, tgt_max_length
        from mytokenizer import load_tokenizer
        
        en_tokenizer = load_tokenizer(en_dir)
        zh_tokenizer = load_tokenizer(zh_dir)
        
        src_max_length = src_max_len
        tgt_max_length = tgt_max_len
        
        jieba.disable_parallel()

    @staticmethod
    def _process_item(item):
        try:
            src_ids = en_tokenizer(
                item['en'],
                truncation=True,
                max_length=src_max_length,
                add_special_tokens=True
            ).input_ids

            zh_words = jieba.lcut(item['zh'])
            tgt = " ".join(zh_words)
            tgt_ids = zh_tokenizer(
                tgt,
                truncation=True,
                max_length=tgt_max_length,
                add_special_tokens=True
            ).input_ids

            if len(src_ids) > src_max_length or len(tgt_ids) > tgt_max_length:
                return None

            return {
                "src": np.array(src_ids, dtype=np.int16),
                "tgt": np.array(tgt_ids[:-1], dtype=np.int16),
                "label": np.array(tgt_ids[1:], dtype=np.int16),
            }
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            return None

    def get_current_file(self):
        return self.parquet_file

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            "src": torch.from_numpy(item["src"]),
            "tgt": torch.from_numpy(item["tgt"]),
            "label": torch.from_numpy(item["label"]),
        }


def collate_fn(batch, pad_token_id: int, src_max_length: int, tgt_max_length: int):
    src = [item["src"].long() for item in batch]
    tgt = [item["tgt"].long() for item in batch]
    label = [item["label"].long() for item in batch]

    # pad src
    src_padded = [F.pad(t, (0, src_max_length - t.size(0)), value=pad_token_id) for t in src]
    src_padded = torch.stack(src_padded)
    
    # pad tgt
    tgt_padded = [F.pad(t, (0, tgt_max_length - t.size(0)), value=pad_token_id) for t in tgt]
    tgt_padded = torch.stack(tgt_padded)
    
    # pad label
    label_padded = [F.pad(t, (0, tgt_max_length - t.size(0)), value=pad_token_id) for t in label]
    label_padded = torch.stack(label_padded)

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "label": label_padded,
    }


def main():
    from torch.utils.data import DataLoader
    from functools import partial
    from mytokenizer import load_tokenizer

    src_max_length = 48
    tgt_max_length = 48
    en_tokenizer_dir = "model/tokenizers/en"
    zh_tokenizer_dir = "model/tokenizers/zh"
    zh_tokenizer = load_tokenizer(zh_tokenizer_dir)
    num_workers = 16

    valid_dir = "data/corpus/parquet/valid/validation-00000-of-00001.parquet"
    valid_cache_dir = "data/cache/valid"
    
    valid_dataset = MyDataset(
        en_tokenizer_dir=en_tokenizer_dir,
        zh_tokenizer_dir=zh_tokenizer_dir,
        parquet_file=valid_dir,
        cache_dir=valid_cache_dir,
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        num_workers=num_workers,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            pad_token_id=zh_tokenizer.pad_token_id,
            src_max_length=src_max_length,
            tgt_max_length=tgt_max_length,
        )
    )

    print("Testing data loading: ")
    for i, batch in enumerate(valid_dataloader):
        print(f"Batch {i}:")
        print(f"src shape: {batch['src'].shape} | sample:\n{batch['src'][0]}")
        print(f"tgt shape: {batch['tgt'].shape} | sample:\n{batch['tgt'][0]}")
        print(f"label shape: {batch['label'].shape} | sample:\n{batch['label'][0]}")
        break

    del valid_dataset, valid_dataloader
    gc.collect()

    train_dir = "data/corpus/parquet/train"
    train_cache_dir = "data/cache/train"

    parquet_files = sorted(glob.glob(os.path.join(train_dir, "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {train_dir}")
    print(f"Found {len(parquet_files)} parquet files to process")
    
    for file_idx, parquet_file in enumerate(parquet_files):
        print(f"\nProcessing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(parquet_file)}")
        
        train_dataset = MyDataset(
            en_tokenizer_dir=en_tokenizer_dir,
            zh_tokenizer_dir=zh_tokenizer_dir,
            parquet_file=parquet_file,
            cache_dir=train_cache_dir,
            src_max_length=src_max_length,
            tgt_max_length=tgt_max_length,
            num_workers=num_workers,
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=partial(
                collate_fn,
                pad_token_id=zh_tokenizer.pad_token_id,
                src_max_length=src_max_length,
                tgt_max_length=tgt_max_length,
            )
        )
        
        print("Testing data loading: ")
        for i, batch in enumerate(train_dataloader):
            print(f"Batch {i}:")
            print(f"src shape: {batch['src'].shape} | sample:\n{batch['src'][0]}")
            print(f"tgt shape: {batch['tgt'].shape} | sample:\n{batch['tgt'][0]}")
            print(f"label shape: {batch['label'].shape} | sample:\n{batch['label'][0]}")
            break
            
        del train_dataset, train_dataloader
        gc.collect()


if __name__ == "__main__":
    main()
