import json
import re


def clean_corpus(input_path, output_path):
    # 定义正则表达式模式
    zh_punctuation_pattern = re.compile(r'[\u3000-\u303F\uFF00-\uFFEF]')  # 匹配所有全角符号
    en_char_pattern = re.compile(r'[A-Za-z]')                             # 匹配英文字符
    en_punctuation_pattern = re.compile(r'[,.;:?!]')                      # 英文标点符号
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cleaned_data = []
    removed_count = 0
    for item in data:
        en = item['en'].strip()
        zh = item['zh'].strip()
        valid = True
        # 处理英文文本
        # 替换中文单引号
        en = en.replace('‘', "'").replace('’', "'")
        
        # 检查中文标点
        if zh_punctuation_pattern.search(en):
            valid = False
            
        # 检查英文结尾符
        if not re.search(r'[.?!]$', en):
            valid = False
        # 处理中文文本
        # 检查英文标点或字符
        if en_punctuation_pattern.search(zh) or en_char_pattern.search(zh):
            valid = False
            
        # 检查中文结尾符
        if not re.search(r'[。？！]$', zh):
            valid = False
        if valid:
            cleaned_data.append({'en': en, 'zh': zh})
        else:
            removed_count += 1
    # 保存清洗结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"清洗完成！\n原始数据：{len(data)}条\n保留数据：{len(cleaned_data)}条\n删除数据：{removed_count}条")


clean_corpus('data/corpus.json', 'data/cleaned_corpus.json')