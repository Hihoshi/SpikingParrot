import json
import jieba
import os
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(
        self, en_tokenizer, zh_tokenizer, corpus_path: str,
        src_max_length: int, tgt_max_length: int, seq_max_length: int
    ):
        """
        - seperate src and tgt language
        - generate seq2seq data and pad
        """
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.seq_max_length = seq_max_length

        # generate cache name
        base_name = os.path.basename(corpus_path)
        cache_dir = os.path.dirname(corpus_path)
        cache_name = f"{base_name}_preprocessed_src{self.src_max_length}_tgt{self.tgt_max_length}.pkl"
        self.cache_path = os.path.join(cache_dir, cache_name)
        
        # create or load tokenized corpus cache file
        if os.path.exists(self.cache_path):
            print(f"loading cached data: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print("creating cache when first loading the dataset, it will take some time...")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            self.processed_data = []
            print(jieba.lcut("这是一个结巴中文分词测试"))
            print("jieba tokenizer test complete")
            for item in tqdm(
                raw_data,
                desc="tokenizing sentences: ",
                unit="pairs",
                colour='green',
                bar_format='{l_bar}{bar:32}{r_bar}',
                dynamic_ncols=True
            ):
                # get src ids
                src_ids = en_tokenizer(
                    item['en'],
                    truncation=True,
                    max_length=self.src_max_length,
                    add_special_tokens=True
                ).input_ids
                
                # get tgt ids
                tgt = " ".join(jieba.lcut(item['zh']))
                tgt_ids = zh_tokenizer(
                    tgt,
                    truncation=True,
                    max_length=self.tgt_max_length,
                    add_special_tokens=True
                ).input_ids
                # [EOS] [BOS] will be automatically added
                # check the length
                if (
                    ((len(src_ids) > self.src_max_length) or (len(tgt_ids) > self.tgt_max_length)) and \
                    (src_ids + tgt_ids < self.seq_max_length)
                ):
                    continue

                self.processed_data.append({
                    "src": src_ids,
                    "tgt": tgt_ids,
                    "label": tgt_ids[1:],
                })

            # save the processed data
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.processed_data, f)
            print(f"cache saved at: {self.cache_path}")

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
    from mytokenizer import load_tokenizer
    from functools import partial

    src_max_length = 48
    tgt_max_length = 36
    seq_max_length = 72

    en_tokenizer = load_tokenizer("model/tokenizers", "en")
    zh_tokenizer = load_tokenizer("model/tokenizers", "zh")
    # create dataset
    corpus_path = "data/corpus.json"
    dataset = MyDataset(
        corpus_path=corpus_path,
        en_tokenizer=en_tokenizer,
        zh_tokenizer=zh_tokenizer,
        src_max_length=src_max_length,
        tgt_max_length=tgt_max_length,
        seq_max_length=seq_max_length,
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
