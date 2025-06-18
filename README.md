# SpikingParrot

从零开始，从数据集处理到一个完整的使用脉冲神经网络的Seq2Seq英译中翻译模型

使用`SpikingLSTM`替换常规`LSTM`，实现了`BiSLSTM`的编码器和`Luong Attention` + `UniSLSTM`的解码器

模型文件大小约**287MB**，以`batch_size=512`，`src_max_length=128`，`tgt_max_length=256`进行`greedy_decode`推理时显存仅占用**5.6G**，最好的检查点稳定BLEU达到**36.51**

## 运行

### 环境

```bash
pip install -r requirements.txt
```

### 数据

`https://huggingface.co/datasets/wmt/wmt19`

下载zh-en数据集文件:

`train-00000-of-00013.parquet` - `train-00012-of-00013.parquet`

`validation-00000-of-00001.parquet`

1. 运行`corpus.py`，分离中英文语料，用于 BPE Tokenizer 的训练，其中中文语料会用 jieba 进行预分词

2. 运行`mytokenizer.py`，训练分词器

   分词器文件保存在`model/tokenizers`

3. 运行`preprocess.py`，预处理数据，分桶保存

   由于原训练数据分片当中数据分布不均匀,需要先合并训练数据，打乱内部句子顺序

   句子会被分词并根据其长度归纳到指定数量的桶当中，桶的划分边界由动态规划最小化总padding数量得出

   例如对应源语句src，当前会被分为20个桶，目标语言tgt也会被分为20个桶，两者取笛卡尔集，最后得到400个桶

   由于翻译任务两者长度强相关，非对角位置的桶样本少且翻译质量相对差，这里设置阈值直接丢弃

   当前参数，总padding率约为**5.39%**，数据丢弃率约为**3.57%**

### 训练

在config当中配置相关信息

运行`train.py`，会根据显存测试最大可用`batch_size`，保存在桶管理器内（需要取消注释`# buckets.reset_batch_size()`）

每个分片训练完成和每轮训练完成都会保存检查点和模型文件同时写入日志

当前参数需要**32G**显存进行混合精度训练，GPU算力**103 Tensor TFLOPS（FP16）**，训练一轮约**5.5H**，训练**4**轮后翻译逐渐正常，最后的检查点在第**10**轮

训练时第一个epoch会从短数据到长数据按桶进行课程训练，后续会打乱桶顺序，由教师强制调度器先下降`teacher_forcing_ratio`然后上升，反复进行

当前检查点文件保存在`temp`目录下，如果需要继续训练，将`temp/latest_checkpoint.pt`复制到`model/checkpoints/`目录下即可

如果复现实验重新训练，保持`model/checkpoints/`目录缺失`latest_checkpoint.pt`文件即可，数据集预处理和训练的随机种子都是**42**，桶管理器的数据当前保存在`temp/buckets_meta.json`中

### 推理

运行`inference.py`计算模型在验证集上BLEU值，使用`greedy_decode`或`beam_decode`（需手动修改相关行的代码）

并写入翻译对照文本到`result/translation_comparison.txt`

### 文件结构

当前由于github的大文件配额限制，已将数据集移除（但保留验证集文件用于快速测试），请自行下载并预处理，相关文件树如下

```bash
├── BucketManager.py
├── MyDataset.py
├── MyLayer.py
├── MyTokenizer.py
├── README.md
├── SpikingParrot.py
├── TryBatch.py
├── corpus.py
├── data
│   ├── cache
│   │   ├── train
│   │   │   ├── buckets_meta.json
│   │   │   ├── cached_src_102_129_tgt_82_99_shard_0.pkl.zst
│   │   │   ├── cached_src_102_129_tgt_99_128_shard_0.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_12_15_shard_0.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_15_18_shard_0.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_2_6_shard_0.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_6_9_shard_0.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_6_9_shard_1.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_9_12_shard_0.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_9_12_shard_1.pkl.zst
│   │   │   ├── cached_src_10_13_tgt_9_12_shard_2.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_12_15_shard_0.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_12_15_shard_1.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_12_15_shard_2.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_15_18_shard_0.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_18_21_shard_0.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_6_9_shard_0.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_9_12_shard_0.pkl.zst
│   │   │   ├── cached_src_13_16_tgt_9_12_shard_1.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_12_15_shard_0.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_12_15_shard_1.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_15_18_shard_0.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_15_18_shard_1.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_15_18_shard_2.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_18_21_shard_0.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_21_24_shard_0.pkl.zst
│   │   │   ├── cached_src_16_19_tgt_9_12_shard_0.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_12_15_shard_0.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_15_18_shard_0.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_15_18_shard_1.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_18_21_shard_0.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_18_21_shard_1.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_21_24_shard_0.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_24_27_shard_0.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_19_22_tgt_9_12_shard_0.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_12_15_shard_0.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_15_18_shard_0.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_18_21_shard_0.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_18_21_shard_1.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_21_24_shard_0.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_21_24_shard_1.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_24_27_shard_0.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_22_25_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_15_18_shard_0.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_18_21_shard_0.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_21_24_shard_0.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_21_24_shard_1.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_24_27_shard_0.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_24_27_shard_1.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_25_28_tgt_33_36_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_15_18_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_18_21_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_21_24_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_24_27_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_24_27_shard_1.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_33_36_shard_0.pkl.zst
│   │   │   ├── cached_src_28_31_tgt_36_40_shard_0.pkl.zst
│   │   │   ├── cached_src_31_34_tgt_18_21_shard_0.pkl.zst
│   │   │   ├── cached_src_31_34_tgt_21_24_shard_0.pkl.zst
│   │   │   ├── cached_src_31_34_tgt_24_27_shard_0.pkl.zst
│   │   │   ├── cached_src_31_34_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_31_34_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_31_34_tgt_33_36_shard_0.pkl.zst
│   │   │   ├── cached_src_31_34_tgt_36_40_shard_0.pkl.zst
│   │   │   ├── cached_src_34_38_tgt_21_24_shard_0.pkl.zst
│   │   │   ├── cached_src_34_38_tgt_24_27_shard_0.pkl.zst
│   │   │   ├── cached_src_34_38_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_34_38_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_34_38_tgt_33_36_shard_0.pkl.zst
│   │   │   ├── cached_src_34_38_tgt_36_40_shard_0.pkl.zst
│   │   │   ├── cached_src_34_38_tgt_40_44_shard_0.pkl.zst
│   │   │   ├── cached_src_38_42_tgt_24_27_shard_0.pkl.zst
│   │   │   ├── cached_src_38_42_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_38_42_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_38_42_tgt_33_36_shard_0.pkl.zst
│   │   │   ├── cached_src_38_42_tgt_36_40_shard_0.pkl.zst
│   │   │   ├── cached_src_38_42_tgt_40_44_shard_0.pkl.zst
│   │   │   ├── cached_src_38_42_tgt_44_49_shard_0.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_2_6_shard_0.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_2_6_shard_1.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_2_6_shard_2.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_2_6_shard_3.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_2_6_shard_4.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_2_6_shard_5.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_6_9_shard_0.pkl.zst
│   │   │   ├── cached_src_3_7_tgt_9_12_shard_0.pkl.zst
│   │   │   ├── cached_src_42_46_tgt_27_30_shard_0.pkl.zst
│   │   │   ├── cached_src_42_46_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_42_46_tgt_33_36_shard_0.pkl.zst
│   │   │   ├── cached_src_42_46_tgt_36_40_shard_0.pkl.zst
│   │   │   ├── cached_src_42_46_tgt_40_44_shard_0.pkl.zst
│   │   │   ├── cached_src_42_46_tgt_44_49_shard_0.pkl.zst
│   │   │   ├── cached_src_42_46_tgt_49_55_shard_0.pkl.zst
│   │   │   ├── cached_src_46_51_tgt_30_33_shard_0.pkl.zst
│   │   │   ├── cached_src_46_51_tgt_33_36_shard_0.pkl.zst
│   │   │   ├── cached_src_46_51_tgt_36_40_shard_0.pkl.zst
│   │   │   ├── cached_src_46_51_tgt_40_44_shard_0.pkl.zst
│   │   │   ├── cached_src_46_51_tgt_44_49_shard_0.pkl.zst
│   │   │   ├── cached_src_46_51_tgt_49_55_shard_0.pkl.zst
│   │   │   ├── cached_src_51_57_tgt_36_40_shard_0.pkl.zst
│   │   │   ├── cached_src_51_57_tgt_40_44_shard_0.pkl.zst
│   │   │   ├── cached_src_51_57_tgt_44_49_shard_0.pkl.zst
│   │   │   ├── cached_src_51_57_tgt_49_55_shard_0.pkl.zst
│   │   │   ├── cached_src_51_57_tgt_55_62_shard_0.pkl.zst
│   │   │   ├── cached_src_57_64_tgt_40_44_shard_0.pkl.zst
│   │   │   ├── cached_src_57_64_tgt_44_49_shard_0.pkl.zst
│   │   │   ├── cached_src_57_64_tgt_49_55_shard_0.pkl.zst
│   │   │   ├── cached_src_57_64_tgt_55_62_shard_0.pkl.zst
│   │   │   ├── cached_src_57_64_tgt_62_71_shard_0.pkl.zst
│   │   │   ├── cached_src_64_73_tgt_44_49_shard_0.pkl.zst
│   │   │   ├── cached_src_64_73_tgt_49_55_shard_0.pkl.zst
│   │   │   ├── cached_src_64_73_tgt_55_62_shard_0.pkl.zst
│   │   │   ├── cached_src_64_73_tgt_62_71_shard_0.pkl.zst
│   │   │   ├── cached_src_64_73_tgt_71_82_shard_0.pkl.zst
│   │   │   ├── cached_src_73_85_tgt_55_62_shard_0.pkl.zst
│   │   │   ├── cached_src_73_85_tgt_62_71_shard_0.pkl.zst
│   │   │   ├── cached_src_73_85_tgt_71_82_shard_0.pkl.zst
│   │   │   ├── cached_src_7_10_tgt_12_15_shard_0.pkl.zst
│   │   │   ├── cached_src_7_10_tgt_2_6_shard_0.pkl.zst
│   │   │   ├── cached_src_7_10_tgt_6_9_shard_0.pkl.zst
│   │   │   ├── cached_src_7_10_tgt_6_9_shard_1.pkl.zst
│   │   │   ├── cached_src_7_10_tgt_6_9_shard_2.pkl.zst
│   │   │   ├── cached_src_7_10_tgt_9_12_shard_0.pkl.zst
│   │   │   ├── cached_src_85_102_tgt_62_71_shard_0.pkl.zst
│   │   │   ├── cached_src_85_102_tgt_71_82_shard_0.pkl.zst
│   │   │   └── cached_src_85_102_tgt_82_99_shard_0.pkl.zst
│   │   ├── txt
│   │   │   ├── corpus.en
│   │   │   └── corpus.zh
│   │   └── valid
│   │       └── cached_src_0_128_tgt_0_128_shard_0.pkl.zst
│   ├── train
│   │   ├── train-00000-of-00013.parquet
│   │   ├── train-00001-of-00013.parquet
│   │   ├── train-00002-of-00013.parquet
│   │   ├── train-00003-of-00013.parquet
│   │   ├── train-00004-of-00013.parquet
│   │   ├── train-00005-of-00013.parquet
│   │   ├── train-00006-of-00013.parquet
│   │   ├── train-00007-of-00013.parquet
│   │   ├── train-00008-of-00013.parquet
│   │   ├── train-00009-of-00013.parquet
│   │   ├── train-00010-of-00013.parquet
│   │   ├── train-00011-of-00013.parquet
│   │   ├── train-00012-of-00013.parquet
│   └── valid
│       └── validation-00000-of-00001.parquet
├── inference.py
├── main.py
├── model
│   ├── checkpoints
│   │   ├── latest_checkpoint.pt
│   │   ├── latest_model.pt
│   └── tokenizers
│       ├── en
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer.json
│       │   └── tokenizer_config.json
│       └── zh
│           ├── special_tokens_map.json
│           ├── tokenizer.json
│           └── tokenizer_config.json
├── preprocess.py
├── requirements.txt
├── results
│   └── translation_comparison.txt
├── temp
│   ├── latest_checkpoint.pt
│   └── latest_model.pt
├── test.py
├── train.py
└── training.log
```
