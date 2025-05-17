# SpikingParrot

从零开始，从数据集处理到一个完整的使用脉冲神经网络的英译中翻译模型

使用SpikingLSTM替换常规LSTM，实现了bislstm的编码器和Luong attention + unislstm的解码器

模型文件大小约320MB，以batch size=128推理时显存仅占用1G，最好的检查点BLEU达到37.8




## 运行

### 必要环境

`pytorch2.1.2+cu121`

```bash
pip install gzip pandas pyarrow jieba transformers tokenizers snntorch tqdm sacrebleu
```


### 准备数据

https://huggingface.co/datasets/wmt/wmt19

下载zh-en数据集文件:

`train-00000-of-00013.parquet` - `train-00012-of-00013.parquet`

`validation-00000-of-00001.parquet`

1. 运行`get_corpus.py`，分离中英文语料，用于 BPE Tokenizer 的训练，其中中文语料会用 jieba 进行预分词

2. 运行`shuffle_parquet.py`
   由于原训练数据分片当中数据分布不均匀

   需要先合并训练数据，打乱内部句子顺序后保存，这里重新切分为16个切片便于后续训练




### 训练分词器（tokenizer）

运行`mytokenizer.py`，训练分词器

分词器文件保存在`model/tokenizers`




### 构建数据集

运行`mydataset.py`

初次运行会缓存分词好的数据在`data/cache/train`和`data/cache/valid`中




### 训练

在config当中配置相关信息

运行`train.py`训练模型

每个分片训练完成和每轮训练完成都会保存检查点和模型文件同时写入日志

当前参数需要**32G**显存进行训练，训练一轮约**8H**，训练**4**轮后翻译逐渐正常

建议最后手动固定 teacher forcing ratio 到 0.8 完成一个分片的训练

teacher forcing ratio 过低会造成训练崩溃，导致翻译文本失去逻辑



### 推理

运行`inference.py`计算模型在验证集上BLEU值

并写入翻译对照文本到`result/translation_comparison.txt`



### 文件结构

相关文件树如下

```bash
├── README.md
├── data
│   ├── cache
│   │   ├── train
│   │   │   ├── cached_0007847e_src48_tgt48.zst
│   │   │   ├── cached_1c9637e7_src48_tgt48.zst
│   │   │   ├── cached_2b6b87e7_src48_tgt48.zst
│   │   │   ├── cached_4dd445b0_src48_tgt48.zst
│   │   │   ├── cached_5343e9ce_src48_tgt48.zst
│   │   │   ├── cached_58b341e6_src48_tgt48.zst
│   │   │   ├── cached_5903783e_src48_tgt48.zst
│   │   │   ├── cached_6188bcf4_src48_tgt48.zst
│   │   │   ├── cached_71bf7b99_src48_tgt48.zst
│   │   │   ├── cached_74a0c991_src48_tgt48.zst
│   │   │   ├── cached_8f1aaed0_src48_tgt48.zst
│   │   │   ├── cached_8f42df53_src48_tgt48.zst
│   │   │   ├── cached_94edf1b7_src48_tgt48.zst
│   │   │   ├── cached_b93dbce4_src48_tgt48.zst
│   │   │   ├── cached_cf7c7f0a_src48_tgt48.zst
│   │   │   └── cached_dfaa20cc_src48_tgt48.zst
│   │   └── valid
│   │       └── cached_2582cbd8_src48_tgt48.zst
│   └── corpus
│       ├── parquet
│       │   ├── train
│       │   │   ├── train_0000.parquet
│       │   │   ├── train_0001.parquet
│       │   │   ├── train_0002.parquet
│       │   │   ├── train_0003.parquet
│       │   │   ├── train_0004.parquet
│       │   │   ├── train_0005.parquet
│       │   │   ├── train_0006.parquet
│       │   │   ├── train_0007.parquet
│       │   │   ├── train_0008.parquet
│       │   │   ├── train_0009.parquet
│       │   │   ├── train_0010.parquet
│       │   │   ├── train_0011.parquet
│       │   │   ├── train_0012.parquet
│       │   │   ├── train_0013.parquet
│       │   │   ├── train_0014.parquet
│       │   │   └── train_0015.parquet
│       │   ├── train_original
│       │   │   ├── train-00000-of-00013.parquet
│       │   │   ├── train-00001-of-00013.parquet
│       │   │   ├── train-00002-of-00013.parquet
│       │   │   ├── train-00003-of-00013.parquet
│       │   │   ├── train-00004-of-00013.parquet
│       │   │   ├── train-00005-of-00013.parquet
│       │   │   ├── train-00006-of-00013.parquet
│       │   │   ├── train-00007-of-00013.parquet
│       │   │   ├── train-00008-of-00013.parquet
│       │   │   ├── train-00009-of-00013.parquet
│       │   │   ├── train-00010-of-00013.parquet
│       │   │   ├── train-00011-of-00013.parquet
│       │   │   ├── train-00012-of-00013.parquet
│       │   └── valid
│       │       └── validation-00000-of-00001.parquet
│       └── txt
│           ├── corpus.en
│           └── corpus.zh
├── get_corpus.py
├── inference.py
├── model
│   ├── checkpoints
│   │   ├── checkpoint_000.pt
│   │   ├── checkpoint_001.pt
│   │   ├── checkpoint_002.pt
│   │   ├── checkpoint_003.pt
│   │   ├── latest_checkpoint.pt
│   │   ├── latest_model.pt
│   │   ├── model_000.pt
│   │   ├── model_001.pt
│   │   ├── model_002.pt
│   │   └── model_003.pt
│   └── tokenizers
│       ├── en
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer.json
│       │   └── tokenizer_config.json
│       └── zh
│           ├── special_tokens_map.json
│           ├── tokenizer.json
│           └── tokenizer_config.json
├── mydataset.py
├── mylayer.py
├── mytokenizer.py
├── results
│   └── translation_comparison.txt
├── shuffle_parquet.py
├── spikingparrot.py
├── train.py
└── training.log
```
