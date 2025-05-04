import pandas as pd
import os
import hashlib  # 用于生成文件唯一标识
from tqdm import tqdm
import jieba
import pickle
import gzip
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
    ):
        self.en_tokenizer_dir = en_tokenizer_dir
        self.zh_tokenizer_dir = zh_tokenizer_dir
        self.parquet_file = parquet_file  # 保存当前处理的文件路径
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.num_workers = num_workers
        
        # 生成基于文件名的缓存路径
        file_hash = hashlib.md5(parquet_file.encode()).hexdigest()[:8]
        cache_name = f"cached_{file_hash}_src{src_max_length}_tgt{tgt_max_length}.pkl"
        self.cache_path = os.path.join(cache_dir, cache_name)
        
        # 检查缓存
        if os.path.exists(self.cache_path):
            print(f"Loading cached data: {os.path.basename(self.cache_path)}")
            with gzip.open(self.cache_path, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print(f"Creating cache from {parquet_file}...")

            if not os.path.exists(parquet_file):
                raise ValueError(f"Parquet file not found: {parquet_file}")

            df = pd.read_parquet(parquet_file)
            raw_data = [{'en': pair['en'], 'zh': pair['zh']} for pair in df['translation'].tolist()]
            
            # 多进程处理
            file_name = os.path.basename(parquet_file)
            self.processed_data = self._parallel_process(raw_data, file_name)
            
            # 手动释放内存
            del df, raw_data
            gc.collect()
            
            # 保存缓存
            print("Saving cache, please wait...")
            with gzip.open(self.cache_path, 'wb') as f:
                pickle.dump(self.processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache saved at: {self.cache_path}")

    def _parallel_process(self, raw_data, file):
        """多进程处理数据的核心方法"""
        num_processes = self.num_workers
        jieba.initialize()
        # 初始化进程池
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
        """子进程初始化函数"""
        global en_tokenizer, zh_tokenizer, src_max_length, tgt_max_length
        from mytokenizer import load_tokenizer
        
        # 加载tokenizer
        en_tokenizer = load_tokenizer(en_dir)
        zh_tokenizer = load_tokenizer(zh_dir)
        
        # 设置长度限制
        src_max_length = src_max_len
        tgt_max_length = tgt_max_len
        
        # 禁用jieba并行（使用多进程替代）
        jieba.disable_parallel()

    @staticmethod
    def _process_item(item):
        """处理单个数据项"""
        try:
            # 处理英文
            src_ids = en_tokenizer(
                item['en'],
                truncation=True,
                max_length=src_max_length,
                add_special_tokens=True
            ).input_ids

            # 处理中文
            zh_words = jieba.lcut(item['zh'])
            tgt = " ".join(zh_words)
            tgt_ids = zh_tokenizer(
                tgt,
                truncation=True,
                max_length=tgt_max_length,
                add_special_tokens=True
            ).input_ids

            # 检查长度
            if len(src_ids) > src_max_length or len(tgt_ids) > tgt_max_length:
                return None

            return {
                # 将列表转换为numpy的int16，占用更少内存
                # 词表大小小于32768都可以
                "src": np.array(src_ids, dtype=np.int16),
                "tgt": np.array(tgt_ids, dtype=np.int16),
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

    # 训练集逐文件处理
    train_dir = "data/corpus/parquet/train"
    train_cache_dir = "data/cache/train"

    # 获取所有parquet文件
    parquet_files = sorted(glob.glob(os.path.join(train_dir, "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {train_dir}")
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # 逐文件处理
    for file_idx, parquet_file in enumerate(parquet_files):
        print(f"\nProcessing file {file_idx+1}/{len(parquet_files)}: {parquet_file}")
        
        # 创建dataset实例
        train_dataset = MyDataset(
            en_tokenizer_dir=en_tokenizer_dir,
            zh_tokenizer_dir=zh_tokenizer_dir,
            parquet_file=parquet_file,
            cache_dir=train_cache_dir,
            src_max_length=src_max_length,
            tgt_max_length=tgt_max_length,
            num_workers=num_workers,
        )
        
        # 创建dataloader
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
        
        # 测试加载
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
