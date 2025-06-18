import json
from typing import List, Optional, Generator, Dict, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import shutil
import math
from MyDataset import MyDataset


class BucketManager:
    def __init__(
        self,
        cache_path: str,
        min_samples: int = 1024,
        max_samples: int = 524288,
        num_cuts: int = 16,
        processed_data: Optional[List[dict]] = None,
        force_rebuild: bool = False,
    ):
        self._validate_init_params(cache_path, processed_data, force_rebuild)

        self.cache_path = Path(cache_path)
        self.num_cuts = num_cuts
        self.min_samples = min_samples
        self.max_samples = max_samples
        
        self.buckets = defaultdict(lambda: defaultdict(list))
        self.valid_buckets = []
        self.total_samples = 0
        self.src_buckets = []
        self.tgt_buckets = []
        
        self.total_original_samples = 0
        self.discarded_samples = 0
        self.total_padding = 0
        self.total_actual_tokens = 0

        if processed_data is not None:
            if not force_rebuild and self._metadata_exists():
                self._load_from_metadata()
            else:
                if force_rebuild and self.cache_path.exists():
                    shutil.rmtree(self.cache_path)
                self.cache_path.mkdir(parents=True, exist_ok=True)
                self._process_data_with_dp(processed_data)
                self._save_metadata()
        else:
            if not self._metadata_exists():
                raise FileNotFoundError(f"No cache found at {cache_path}")
            self._load_from_metadata()

    def _validate_init_params(
        self,
        cache_path: str,
        processed_data: Optional[List[dict]],
        force_rebuild: bool
    ):
        if not cache_path:
            raise ValueError("cache_path cannot be empty")
            
        if processed_data is None and not force_rebuild and not Path(cache_path).exists():
            raise FileNotFoundError(f"Cache path missing: {cache_path}")

    @staticmethod
    def _optimal_1d_partition(lengths: List[int], num_cuts: int) -> List[Tuple[int, int]]:
        if not lengths:
            return []

        length_counts = Counter(lengths)
        unique_lengths = sorted(length_counts.keys())
        
        if len(unique_lengths) <= num_cuts:
            buckets = []
            for i, length in enumerate(unique_lengths):
                start = length
                end = unique_lengths[i + 1] if i + 1 < len(unique_lengths) else length + 1
                buckets.append((start, end))
            return buckets
        
        n = len(unique_lengths)
        
        dp = [[float('inf')] * (num_cuts + 1) for _ in range(n + 1)]
        parent = [[-1] * (num_cuts + 1) for _ in range(n + 1)]
        
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i + 1, num_cuts + 1)):
                for k in range(j - 1, i):
                    if dp[k][j - 1] == float('inf'):
                        continue
                    
                    bucket_start = unique_lengths[k]
                    bucket_end = unique_lengths[i - 1] + 1
                    bucket_max = unique_lengths[i - 1]
                    
                    padding_in_bucket = 0
                    for idx in range(k, i):
                        length = unique_lengths[idx]
                        count = length_counts[length]
                        padding_in_bucket += (bucket_max - length) * count
                    
                    total_padding = dp[k][j - 1] + padding_in_bucket
                    
                    if total_padding < dp[i][j]:
                        dp[i][j] = total_padding
                        parent[i][j] = k
        
        if dp[n][num_cuts] == float('inf'):
            return [(min(lengths), max(lengths) + 1)]
        
        buckets = []
        i, j = n, num_cuts
        
        while j > 0 and i > 0:
            k = parent[i][j]
            if k < 0 or k >= i:
                break
            bucket_start = unique_lengths[k]
            bucket_end = unique_lengths[i - 1] + 1
            buckets.append((bucket_start, bucket_end))
            i, j = k, j - 1
        
        buckets.reverse()
        return buckets

    def _create_2d_buckets(
        self,
        data: List[Tuple[int, int]]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if not data:
            return []
        
        src_lengths = [item[0] for item in data]
        tgt_lengths = [item[1] for item in data]
        
        print("Calculating optimal buckets for src lengths...")
        self.src_buckets = self._optimal_1d_partition(src_lengths, self.num_cuts)
        print(f"src buckets: {self.src_buckets}")
        
        print("Calculating optimal buckets for tgt lengths...")
        self.tgt_buckets = self._optimal_1d_partition(tgt_lengths, self.num_cuts)
        print(f"tgt buckets: {self.tgt_buckets}")
        
        bucket_samples = {}
        for src_bucket in self.src_buckets:
            for tgt_bucket in self.tgt_buckets:
                bucket_key = (src_bucket, tgt_bucket)
                bucket_samples[bucket_key] = []
        
        for src_len, tgt_len in data:
            src_bucket = None
            tgt_bucket = None
            
            for bucket in self.src_buckets:
                if bucket[0] <= src_len < bucket[1]:
                    src_bucket = bucket
                    break
            
            for bucket in self.tgt_buckets:
                if bucket[0] <= tgt_len < bucket[1]:
                    tgt_bucket = bucket
                    break
            
            if src_bucket and tgt_bucket:
                bucket_key = (src_bucket, tgt_bucket)
                bucket_samples[bucket_key].append((src_len, tgt_len))
        
        valid_buckets = []
        
        for bucket_key, samples in bucket_samples.items():
            if len(samples) >= self.min_samples:
                valid_buckets.append(bucket_key)
        
        return valid_buckets

    def _process_data_with_dp(self, data: List[dict]):
        length_pairs = [(len(item["src"]), len(item["tgt"])) for item in data]
        self.total_original_samples = len(data)
        
        valid_buckets = self._create_2d_buckets(length_pairs)
        
        bucket_data = {bucket: [] for bucket in valid_buckets}

        total_actual_tokens = 0
        total_padding = 0
        used_samples = 0
        
        for item in data:
            src_len = len(item["src"])
            tgt_len = len(item["tgt"])
            found_bucket = False
            
            for (src_start, src_end), (tgt_start, tgt_end) in valid_buckets:
                if src_start <= src_len < src_end and tgt_start <= tgt_len < tgt_end:
                    src_max = src_end - 1
                    tgt_max = tgt_end - 1
                    src_pad = src_max - src_len
                    tgt_pad = tgt_max - tgt_len
                    
                    total_padding += src_pad + tgt_pad
                    total_actual_tokens += src_len + tgt_len
                    used_samples += 1
                    
                    bucket_data[(src_start, src_end), (tgt_start, tgt_end)].append(item)
                    found_bucket = True
                    break
            
            if not found_bucket:
                pass
        
        self.discarded_samples = len(data) - used_samples
        self.total_padding = total_padding
        self.total_actual_tokens = total_actual_tokens
        
        print("\nBuilding datasets from buckets...")
        total_samples = 0
        self.valid_buckets = []
        
        sorted_buckets = sorted(valid_buckets, key=lambda x: (x[1][0], x[0][0]))
        
        for bucket_key in sorted_buckets:
            (src_start, src_end), (tgt_start, tgt_end) = bucket_key
            bucket_items = bucket_data[bucket_key]
            data_len = len(bucket_items)
            
            base_num_shards = max(1, (data_len + self.max_samples - 1) // self.max_samples)
            
            last_shard_size = data_len % self.max_samples
            
            if last_shard_size == 0 and data_len > 0:
                last_shard_size = self.max_samples
            
            if base_num_shards > 1 and last_shard_size <= self.max_samples * 0.5:
                num_shards = base_num_shards - 1
            else:
                num_shards = base_num_shards
            
            for shard_idx in range(num_shards):
                if shard_idx == num_shards - 1:
                    start_idx = shard_idx * self.max_samples
                    end_idx = data_len
                else:
                    start_idx = shard_idx * self.max_samples
                    end_idx = start_idx + self.max_samples
                
                shard_data = bucket_items[start_idx:end_idx]
                shard_size = len(shard_data)
                
                MyDataset(
                    cache_path=str(self.cache_path),
                    processed_data=shard_data,
                    src_range=(src_start, src_end),
                    tgt_range=(tgt_start, tgt_end),
                    shard_idx=shard_idx,
                )
                
                bucket_info = {
                    "src_range": (src_start, src_end),
                    "tgt_range": (tgt_start, tgt_end),
                    "shard_idx": shard_idx,
                    "num_shards": num_shards,
                    "suggested_batch_size": 0,
                    "num_samples": shard_size
                }
                self.valid_buckets.append(bucket_info)
                total_samples += shard_size
        self.total_samples = total_samples
        print(f"Valid buckets: {len(self.valid_buckets)}")
        print(f"Total samples: {total_samples}")

    def _save_metadata(self):
        meta = {
            "num_cuts": self.num_cuts,
            "valid_buckets": [{
                "src_range": list(b["src_range"]),
                "tgt_range": list(b["tgt_range"]),
                "shard_idx": b["shard_idx"],
                "num_shards": b["num_shards"],
                "suggested_batch_size": b["suggested_batch_size"],
                "num_samples": b["num_samples"]
            } for b in self.valid_buckets],
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "total_samples": self.total_samples,
            "total_original_samples": self.total_original_samples,
            "discarded_samples": self.discarded_samples,
            "total_padding": self.total_padding,
            "total_actual_tokens": self.total_actual_tokens
        }

        meta_path = self.cache_path / "buckets_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Metadata saved to {meta_path}")

    def _load_from_metadata(self):
        meta_path = self.cache_path / "buckets_meta.json"
        meta = json.loads(meta_path.read_text())

        self.num_cuts = meta.get("num_cuts", 8)
        self.min_samples = meta.get("min_samples", self.min_samples)
        self.max_samples = meta.get("max_samples", self.max_samples)
        self.total_samples = meta.get("total_samples", 0)
        
        self.total_original_samples = meta.get("total_original_samples", 0)
        self.discarded_samples = meta.get("discarded_samples", 0)
        self.total_padding = meta.get("total_padding", 0)
        self.total_actual_tokens = meta.get("total_actual_tokens", 0)

        self.valid_buckets = []
        for b in meta["valid_buckets"]:
            self.valid_buckets.append({
                "src_range": tuple(b["src_range"]),
                "tgt_range": tuple(b["tgt_range"]),
                "shard_idx": b["shard_idx"],
                "num_shards": b["num_shards"],
                "suggested_batch_size": b.get("suggested_batch_size", 0),
                "num_samples": b["num_samples"]
            })

        print(f"Loaded {len(self.valid_buckets)} buckets from {meta_path}")

    def _metadata_exists(self) -> bool:
        return (self.cache_path / "buckets_meta.json").exists()

    def __iter__(self) -> Generator[Dict, None, None]:
        yield from self.valid_buckets
    
    def __len__(self) -> int:
        return len(self.valid_buckets)
    
    def __getitem__(self, index: int) -> dict:
        return self.valid_buckets[index]

    def reset_batch_size(self) -> None:
        for b in self.valid_buckets:
            b["suggested_batch_size"] = 0
    
    def find_optimal_batch_size(self, find_batch_size_func) -> None:
        bucket_type_map = {}
        for i, bucket in enumerate(self.valid_buckets):
            src_range = bucket["src_range"]
            tgt_range = bucket["tgt_range"]
            bucket_type = (src_range, tgt_range)
            
            if bucket_type in bucket_type_map:
                bucket["suggested_batch_size"] = bucket_type_map[bucket_type]
                print(f"Bucket {i + 1}/{len(self.valid_buckets)} reused batch size: {bucket['suggested_batch_size']}\n")
                continue
            
            print(
                f"Searching optimal batch size for bucket {i + 1}/{len(self.valid_buckets)} " + \
                f"src: {src_range} tgt: {tgt_range} ..."
            )
            
            batch_size = find_batch_size_func(src_max=src_range[1], tgt_max=tgt_range[1])
            bucket["suggested_batch_size"] = batch_size
            bucket_type_map[bucket_type] = bucket["suggested_batch_size"]
            
            print(f"Found batch size: {bucket['suggested_batch_size']}\n")
        
        self._save_metadata()
    
    def found_optimal(self) -> bool:
        return all(b["suggested_batch_size"] > 0 for b in self.valid_buckets)
    
    def get_total_iterations(self, safety_factor: float, drop_last: bool = False) -> int:
        total_iterations = 0
        
        for bucket in self.valid_buckets:
            batch_size = math.floor(bucket["suggested_batch_size"] * safety_factor)
            num_samples = bucket["num_samples"]
            
            if batch_size <= 0:
                raise ValueError("Batch size must be positive. Call find_optimal_batch_size first.")
                
            if drop_last:
                iterations = num_samples // batch_size
            else:
                iterations = (num_samples + batch_size - 1) // batch_size
            
            total_iterations += iterations
        
        return total_iterations

    def get_info(self) -> List[dict]:
        return self.valid_buckets

    def print_stats(self):
        print("\n" + "=" * 60)
        print("Bucket Statistics")
        print("=" * 60)
        print(f"Total buckets: {len(self.valid_buckets)}")
        print(f"Total samples: {self.total_samples}")
        print(f"Min samples per bucket: {self.min_samples}")
        print(f"Max samples per bucket: {self.max_samples}")
        
        print("\nData Distribution:")
        print("-" * 60)
        print(f"Original samples: {self.total_original_samples}")
        print(f"Discarded samples: {self.discarded_samples} ({self.discarded_samples / self.total_original_samples * 100:.2f}%)")
        
        total_tokens_with_padding = self.total_actual_tokens + self.total_padding
        if total_tokens_with_padding > 0:
            padding_rate = self.total_padding / total_tokens_with_padding
        else:
            padding_rate = 0.0
        print(f"Total padding tokens: {self.total_padding}")
        print(f"Padding rate: {padding_rate * 100:.2f}%")
        
        print("\nBucket Details:")
        print("-" * 80)
        print(f"{'Bucket ID':<8} {'Src Range':<15} {'Tgt Range':<15} {'Samples':<10} {'Shards':<8} {'Batch Size':<10}")
        print("-" * 80)
        
        for i, bucket in enumerate(self.valid_buckets):
            src_range = f"{bucket['src_range'][0]}-{bucket['src_range'][1] - 1}"
            tgt_range = f"{bucket['tgt_range'][0]}-{bucket['tgt_range'][1] - 1}"
            samples = bucket['num_samples']
            shards = f"{bucket['shard_idx'] + 1}/{bucket['num_shards']}"
            batch_size = bucket['suggested_batch_size']
            
            print(f"{i:<8} {src_range:<15} {tgt_range:<15} {samples:<10} {shards:<8} {batch_size:<10}")
