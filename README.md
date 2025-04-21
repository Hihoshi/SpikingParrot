# SpikingParrot
从零开始，数据集到一个完整的使用脉冲神经网络的Seq2Seq翻译模型，适合机器学习流程的熟悉


## 运行
### 必要环境
pytorch2.1.2+cu121
`pip install jieba transformers tokenizers snntorch tqdm`

### 准备数据

`corpus.json`存储双语数据集，示例格式如下：

```json
{
    {
        "zh": "表演的明星是X女孩团队——由一对具有天才技艺的艳舞女孩们组成，其中有些人受过专业的训练。",
        "en": "the show stars the X Girls - a troupe of talented topless dancers , some of whom are classically trained ."
    },
    {
        "zh": "表演的压轴戏是闹剧版《天鹅湖》，男女小人们身着粉红色的芭蕾舞裙扮演小天鹅。",
        "en": "the centerpiece of the show is a farcical rendition of Swan Lake in which male and female performers dance in pink tutus and imitate swans ."
    },
}
```
运行`get_corpus.py`，分离中英文语料

### 训练分词器（tokenizer）

运行`mytokenizer.py`，训练分词器：


### 构建数据集

运行`mydataset.py`，初次运行会储存分词好的缓存

### 训练

运行`train.py`训练模型即可训练模型

