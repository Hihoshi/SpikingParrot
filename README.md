# T-SLSTM



## 运行

### 准备数据

`corpus.json`存储数据集，格式如下

```json
{
    "name": "ccmt",
    "len": 1049989,
    "pairs": [
        {
            "idx": 0,
            "zh": "表演的明星是X女孩团队——由一对具有天才技艺的艳舞女孩们组成，其中有些人受过专业的训练。",
            "en": "the show stars the X Girls - a troupe of talented topless dancers , some of whom are classically trained ."
        },
        {
            "idx": 1,
            "zh": "表演的压轴戏是闹剧版《天鹅湖》，男女小人们身着粉红色的芭蕾舞裙扮演小天鹅。",
            "en": "the centerpiece of the show is a farcical rendition of Swan Lake in which male and female performers dance in pink tutus and imitate swans ."
        },
    ]
}
```

### 训练分词器（tokenizer）

运行`tokenizer.py`，训练分词器，训练完成后注释掉训练代码，使用`json`装载分词器：

```python
# load pretrained tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer = tokenizer.from_file("BPE_tokenizer.json")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    eos_token="[EOS]",
    pad_token="[PAD]"
)
```

### 构建数据集

运行`mydataset.py`

`MyDataset("corpus.json", use_cached=False)`得到缓存文件后

再使用`MyDataset("cached.json", use_cached=True)`加快数据集加载速度（10x）

### 训练

运行`train.py`训练模型，会`model/translation/`下生成

`log.csv`训练日志，`net.pth`模型文件，`optimizer.pth`优化器文件

