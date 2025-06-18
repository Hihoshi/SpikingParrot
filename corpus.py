import os
import pandas as pd
import glob
import jieba
from multiprocessing import Pool
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

NUM_WORKERS = 20
INPUT_DIRS = {
    'train': 'data/train',
    'valid': 'data/valid'
}
OUTPUT_DIR = 'data/cache/txt'
EN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'corpus.en')
ZH_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'corpus.zh')


def init_jieba():
    jieba_logger = logging.getLogger('jieba')
    jieba_logger.setLevel(logging.WARNING)
    jieba.disable_parallel()


def process_line(record):
    try:
        en_text = record['translation']['en']
        zh_text = record['translation']['zh']
        
        # 中文分词
        zh_words = jieba.lcut(zh_text)
        zh_sentence = ' '.join(zh_words)
        
        return (en_text, zh_sentence)
    except KeyError as e:
        logging.warning(f"Missing field in record: {str(e)}")
        return None
    except Exception as e:
        logging.warning(f"Line processing error: {str(e)}")
        return None


def process_shard(shard_path):
    try:
        df = pd.read_parquet(shard_path)
        records = df.to_dict(orient='records')
        total = len(records)
        logging.info(f"Processing {shard_path} ({total} lines)")
        
        with Pool(NUM_WORKERS, initializer=init_jieba) as pool:
            results = []
            for result in tqdm(
                pool.imap(process_line, records),
                total=total,
                desc=f"Processing {os.path.basename(shard_path)}",
                unit="lines",
                colour='green',
                bar_format='{l_bar}{bar:32}{r_bar}'
            ):
                if result is not None:
                    results.append(result)
        
        en_sentences, zh_sentences = zip(*results) if results else ([], [])
        logging.info(f"Processed {len(results)} lines from {shard_path}")
        return list(en_sentences), list(zh_sentences)
        
    except Exception as e:
        logging.error(f"Shard processing failed: {shard_path} - {str(e)}")
        return [], []


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_shards = []
    for _, dir_path in INPUT_DIRS.items():
        shards = glob.glob(os.path.join(dir_path, '*.parquet'))
        if not shards:
            logging.warning(f"No Parquet files found in {dir_path}")
        all_shards.extend(shards)
    
    if not all_shards:
        logging.error("No Parquet files found in any input directories")
        return
    
    all_shards.sort(key=lambda x: os.path.abspath(x))
    
    logging.info(f"Found {len(all_shards)} shards to process")
    jieba.initialize()

    all_en = []
    all_zh = []
    
    for shard_path in all_shards:
        en_sentences, zh_sentences = process_shard(shard_path)
        all_en.extend(en_sentences)
        all_zh.extend(zh_sentences)
    
    if len(all_en) != len(all_zh):
        logging.warning(f"Data length mismatch: {len(all_en)} English vs {len(all_zh)} Chinese sentences")
    
    logging.info(f"Writing {len(all_en)} sentences to final files")
    
    with open(EN_OUTPUT_PATH, 'w', encoding='utf-8') as f_en, \
            open(ZH_OUTPUT_PATH, 'w', encoding='utf-8') as f_zh:
        
        for en, zh in tqdm(zip(all_en, all_zh), total=len(all_en), desc="Writing files"):
            f_en.write(en + '\n')
            f_zh.write(zh + '\n')
    
    logging.info("Corpus generation completed successfully")
    logging.info(f"English corpus saved at: {EN_OUTPUT_PATH}")
    logging.info(f"Chinese corpus saved at: {ZH_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
