import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def merge_shuffle_split_parquet(input_dir, output_dir, num_shards=16):
    # aquire all parquet files
    files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    # read through pyarrow
    dataset = pq.ParquetDataset(files)
    table = dataset.read()
    
    # transform to pandas dataframe
    df = table.to_pandas()
    
    # ensure translation includes both 'en' and 'zh'
    if not all(len(row) == 2 and 'en' in row and 'zh' in row for row in df['translation']):
        raise ValueError("inconsistent data detected, please verify")
    
    # shuffle all data
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # create now shards
    shards = np.array_split(shuffled_df, num_shards)
    
    # create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # save shards
    for i, shard in enumerate(shards):
        output_path = os.path.join(output_dir, f"train_{i:04d}.parquet")
        pq.write_table(
            pa.Table.from_pandas(shard),
            output_path,
            compression='snappy',
            use_dictionary=['translation']
        )
    
    print(f"successfully generate {num_shards} shards, total {len(shuffled_df)} samples")


merge_shuffle_split_parquet("data/corpus/parquet/train", "data/corpus/parquet/train_shuffled", 16)
