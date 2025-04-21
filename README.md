# SpikingParrot
一个使用脉冲长短期神经网络构成的Seq2Seq翻译模型

## 运行

### 准备数据

`corpus.json`存储数据集，格式如下

```json
{
    "pairs": [
        {
            "zh": "表演的明星是X女孩团队——由一对具有天才技艺的艳舞女孩们组成，其中有些人受过专业的训练。",
            "en": "the show stars the X Girls - a troupe of talented topless dancers , some of whom are classically trained ."
        },
        {
            "zh": "表演的压轴戏是闹剧版《天鹅湖》，男女小人们身着粉红色的芭蕾舞裙扮演小天鹅。",
            "en": "the centerpiece of the show is a farcical rendition of Swan Lake in which male and female performers dance in pink tutus and imitate swans ."
        },
    ]
}
```

### 训练分词器（tokenizer）

运行`tokenizer.py`，训练分词器

### 构建数据集

运行`mydataset.py`

首次运行会得到分词好之后的缓存文件

### 训练

运行`train.py`训练模型，会`model/`下生成
