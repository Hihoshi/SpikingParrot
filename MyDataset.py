import os
import pickle
import zstandard as zstd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        processed_data: list[dict] = None,
        src_range: tuple = (0, 0),
        tgt_range: tuple = (0, 0),
        shard_idx: int = 0,
        zstd_level: int = 9
    ):
        self.src_range = src_range
        self.tgt_range = tgt_range
        self.zstd_level = zstd_level
        self.processed_data = processed_data

        base_name = f"cached_src_{src_range[0]}_{src_range[1]}_tgt_{tgt_range[0]}_{tgt_range[1]}"
        self.cache_path = os.path.join(
            cache_path,
            f"{base_name}_shard_{shard_idx}.pkl.zst"
        )

        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    self.processed_data = pickle.load(reader)
            return
        elif self.processed_data is None:
            raise ValueError(f"Cache file not found: {self.cache_path} and no data provided")

        # Cache保存
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

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            "src": torch.from_numpy(item["src"]).long(),
            "tgt": torch.from_numpy(item["tgt"]).long(),
            "label": torch.from_numpy(item["label"]).long(),
        }


def collate_fn(batch, pad_token_id: int, src_max_length: int, tgt_max_length: int):
    src = [item["src"] for item in batch]
    tgt = [item["tgt"] for item in batch]
    label = [item["label"] for item in batch]

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
