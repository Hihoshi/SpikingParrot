import json
from openai import OpenAI

client = OpenAI(
    # base_url="http://localhost:11434/v1",  # Ollama 的 API 端点
    # api_key="ollama"
    # base_url="https://api.deepseek.com/v1",
    # api_key="sk-965f71fd8b4c42c39c3f22b18b4dcc91"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-7d5f13ee521c47baa7cfb14e94cd3e03"
)

with open('data/cleaned_corpus.json', 'r', encoding='utf8') as f:
    corpus = json.load(f)

for i in corpus:
    msg = '''
        我在创建一个机器翻译的数据集，我会给你一个中英句对，你需要输出修改的后的句子：
        + 检查格式错误
            + 中文句子使用中文标点，英文使用英文标点，尤其注意中英文引号的区别
            + 英文的句首大小写
        + 检查翻译是否错误
            + 如果翻译大致没有问题，就保持原本的句子，如果有非常明显的翻译问题则进行修改，但是同样以最少修改为原则，
        + 输出格式
            + 先输出\"英文:\"和中文句子，换行，再输出\"中文:\"和英文句子
            + 不要输出无关内容，不需要对修改做出说明
        以下是中英句对：{}
        '''.format("英文:" + i["en"] + "\n" + "中文:" + i["zh"] + "\n",)
    completion = client.chat.completions.create(
        # model="qwen2.5:7b",
        # model="deepseek-chat",
        model="qwen-plus",
        messages=[
            {
                "role": "user",
                "content": msg
            }
        ],
        stream=True
    )

    full_content = ""
    for chunk in completion:
        # 如果stream_options.include_usage为True，则最后一个chunk的choices字段为空列表，需要跳过（可以通过chunk.usage获取 Token 使用量）
        full_content += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    with open("data/corpus.txt", "a", encoding="utf8") as f:
        f.write(full_content + "\n\n")