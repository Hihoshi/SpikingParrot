import json
import jieba
import os
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class MyDataset(Dataset):
    def __init__(
        self, en_tokenizer_dir: str, zh_tokenizer_dir: str, corpus_path: str,
        src_max_length: int, tgt_max_length: int, num_workers: int = None,
    ):
        self.en_tokenizer_dir = en_tokenizer_dir
        self.zh_tokenizer_dir = zh_tokenizer_dir
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.num_workers = num_workers if num_workers is not None else cpu_count()

        base_name = os.path.basename(corpus_path)
        cache_dir = os.path.dirname(corpus_path)
        cache_name = f"{base_name}_preprocessed_src{self.src_max_length}_tgt{self.tgt_max_length}.pkl"
        self.cache_path = os.path.join(cache_dir, cache_name)
        
        if os.path.exists(self.cache_path):
            print(f"loading cached data: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print("creating cache when first loading the dataset, it will take some time...")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 多进程处理
            print(jieba.lcut("这是一个结巴中文分词测试"))
            print("jieba pretokenizer test complete")
            self.processed_data = self._parallel_process(raw_data)
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.processed_data, f)
            print(f"cache saved at: {self.cache_path}")

    def _parallel_process(self, raw_data):
        """多进程处理数据的核心方法"""
        num_processes = self.num_workers
        
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
                desc="Tokenizing sentences",
                unit="pairs",
                colour='green',
                bar_format='{l_bar}{bar:32}{r_bar}',
                dynamic_ncols=True
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
                "src": src_ids,
                "tgt": tgt_ids,
                "label": tgt_ids[1:],
            }
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            return None

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch, pad_token_id: int, src_max_length: int, tgt_max_length: int):
    src = [torch.tensor(item["src"]) for item in batch]
    tgt = [torch.tensor(item["tgt"]) for item in batch]
    label = [torch.tensor(item["label"]) for item in batch]

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
    tgt_max_length = 32

    en_tokenizer_dir = "model/tokenizers/en"
    zh_tokenizer_dir = "model/tokenizers/zh"

    zh_tokenizer = load_tokenizer(zh_tokenizer_dir)

    # create dataset
    corpus_path = "data/corpus.json"
    dataset = MyDataset(
        corpus_path=corpus_path,
        en_tokenizer_dir=en_tokenizer_dir,
        zh_tokenizer_dir=zh_tokenizer_dir,
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        num_workers=8,
    )
    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            pad_token_id=zh_tokenizer.pad_token_id,
            src_max_length=src_max_length,
            tgt_max_length=tgt_max_length,
        )
    )
    # load test sample
    print("test data loading: ")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"src shape: {batch['src'].shape} | sample:\n{batch['src'][0]}")
        print(f"tgt shape: {batch['tgt'].shape} | sample:\n{batch['tgt'][0]}")
        print(f"label shape: {batch['label'].shape} | sample:\n{batch['label'][0]}")
        break


if __name__ == "__main__":
    main()
