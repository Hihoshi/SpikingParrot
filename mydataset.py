import json
import os
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, tokenizer, corpus_path: str, src_max_length: int, tgt_max_length: int, seq_max_length: int):
        """
        - 分离源语言(src)和目标语言(tgt)
        - 动态生成逐步解码的上下文
        """
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.seq_max_length = seq_max_length

        # 生成唯一缓存文件名（包含长度信息）
        base_name = os.path.basename(corpus_path)
        cache_dir = os.path.dirname(corpus_path)
        cache_name = f"{base_name}_preprocessed_src{self.src_max_length}_tgt{self.tgt_max_length}.pkl"
        self.cache_path = os.path.join(cache_dir, cache_name)
        
        # 加载或创建预处理数据
        if os.path.exists(self.cache_path):
            print(f"加载预处理缓存: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print("创建预处理缓存，首次运行需要较长时间...")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            self.processed_data = []

            for item in tqdm(
                raw_data,
                desc="预处理语料",
                unit="样本",
                colour='green',
                bar_format='{l_bar}{bar:32}{r_bar}',
                dynamic_ncols=True
            ):
                # 编码源语言（英语）
                src_ids = tokenizer(
                    item['en'],
                    truncation=True,
                    max_length=self.src_max_length,
                    add_special_tokens=True
                ).input_ids
                
                # 编码目标语言（中文）
                tgt_ids = tokenizer(
                    item['zh'],
                    truncation=True,
                    max_length=self.tgt_max_length,
                    add_special_tokens=True
                ).input_ids
                # 已经在中文和英文句子中自动添加[BOS]和[BOS]，所以这里不需要手动添加。
                # 检查长度有效性
                if (
                    ((len(src_ids) > self.src_max_length) or (len(tgt_ids) > self.tgt_max_length)) and
                    (src_ids + tgt_ids < self.seq_max_length)
                ):
                    continue

                self.processed_data.append({
                    "src": src_ids,
                    "tgt": tgt_ids,
                    "label": tgt_ids[1:],
                })
            
            # 保存预处理结果
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.processed_data, f)
            print(f"缓存已保存至: {self.cache_path}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch, pad_token_id: int, src_max_length: int, tgt_max_length: int):
    src = [torch.tensor(item["src"]) for item in batch]
    tgt = [torch.tensor(item["tgt"]) for item in batch]
    label = [torch.tensor(item["label"]) for item in batch]

    # 源序列填充
    src_padded = [F.pad(t, (0, src_max_length - t.size(0)), value=pad_token_id) for t in src]
    src_padded = torch.stack(src_padded)
    
    # 目标序列填充
    tgt_padded = [F.pad(t, (0, tgt_max_length - t.size(0)), value=pad_token_id) for t in tgt]
    tgt_padded = torch.stack(tgt_padded)
    
    # 标签处理
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
    tokenizer = load_tokenizer("model/tokenizers")

    # 创建数据集实例（使用示例参数）
    corpus_path = "data/corpus.json"
    dataset = MyDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        src_max_length=48,
        tgt_max_length=32,
        seq_max_length=64,
    )
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            pad_token_id=tokenizer.pad_token_id,
            src_max_length=48,
            tgt_max_length=32,
        )
    )
    # 测试数据加载
    print("\n测试数据加载：")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"src形状: {batch['src'].shape} | 内容示例:\n{batch['src'][0]}")
        print(f"tgt形状: {batch['tgt'].shape} | 内容示例:\n{batch['tgt'][0]}")
        print(f"label形状: {batch['label'].shape} | 内容示例:\n{batch['label'][0]}")
        break


if __name__ == "__main__":
    main()
