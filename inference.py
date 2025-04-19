import torch
from model import SpikingSeq2Seq
from mytokenizer import load_tokenizer
import argparse


def load_model(model_path, tokenizer_path, max_length, device):
    """加载训练好的模型和分词器"""
    tokenizer = load_tokenizer(tokenizer_path)
    model = SpikingSeq2Seq(
        vocab_size=tokenizer.vocab_size,
        max_len=max_length,
        embedding_dim=2048,
        time_step=32,
        hidden_dim=64,
        output_dim=128
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, tokenizer


def greedy_decode(model, tokenizer, en_text, max_length, device):
    """贪婪解码实现"""
    model.eval()
    with torch.no_grad():
        # 预处理英文输入
        en_encoding = tokenizer(
            en_text,
            truncation=True,
            max_length=max_length - 1,  # 为BOS留空间
            add_special_tokens=True
        )
        en_ids = en_encoding.input_ids
        current_zh_ids = [tokenizer.bos_token_id]
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        max_zh_length = max_length - len(en_ids)
        if max_zh_length <= 0:
            return tokenizer.decode([eos_token_id], skip_special_tokens=True)

        # 自回归生成循环
        for _ in range(max_zh_length - 1):
            # 构建当前上下文
            current_context = en_ids + current_zh_ids
            if len(current_context) > max_length:
                current_context = current_context[:max_length]
            else:
                current_context += [pad_token_id] * (max_length - len(current_context))
            
            # 模型推理
            context_tensor = torch.tensor([current_context], device=device)
            logits = model(context_tensor)
            
            # 选择概率最高的token
            next_token = logits.argmax(dim=-1).item()
            
            # 终止条件检查
            if next_token == eos_token_id:
                break
                
            current_zh_ids.append(next_token)

        # 添加终止符（如果需要）
        if current_zh_ids[-1] != eos_token_id:
            current_zh_ids.append(eos_token_id)

        return tokenizer.decode(current_zh_ids, skip_special_tokens=True)


def beam_search_decode(model, tokenizer, en_text, max_length, device, beam_width=5):
    """束搜索解码实现"""
    model.eval()
    with torch.no_grad():
        # 预处理英文输入
        en_encoding = tokenizer(
            en_text,
            truncation=True,
            max_length=max_length - 1,
            add_special_tokens=True
        )
        en_ids = en_encoding.input_ids
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        max_zh_length = max_length - len(en_ids)
        if max_zh_length <= 0:
            return tokenizer.decode([eos_token_id], skip_special_tokens=True)

        # 初始化束搜索
        beam = [{
            'tokens': [bos_token_id],
            'score': 0.0,
            'finished': False
        }]

        # 束搜索迭代
        for _ in range(max_zh_length - 1):
            candidates = []
            active_beam = [b for b in beam if not b['finished']]
            
            if not active_beam:
                break

            # 批量处理活动束
            batch_contexts = []
            for item in active_beam:
                current_context = en_ids + item['tokens']
                if len(current_context) > max_length:
                    current_context = current_context[:max_length]
                else:
                    current_context += [pad_token_id] * (max_length - len(current_context))
                batch_contexts.append(current_context)

            # 模型推理
            context_tensor = torch.tensor(batch_contexts, device=device)
            logits = model(context_tensor)
            probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # 生成候选
            for i, item in enumerate(active_beam):
                top_probs, top_indices = torch.topk(probs[i], beam_width)
                for j in range(beam_width):
                    new_tokens = item['tokens'].copy()
                    new_score = item['score'] + top_probs[j].item()
                    new_token = top_indices[j].item()
                    
                    new_tokens.append(new_token)
                    is_finished = (new_token == eos_token_id) or (len(new_tokens) >= max_zh_length)
                    
                    candidates.append({
                        'tokens': new_tokens,
                        'score': new_score,
                        'finished': is_finished
                    })

            # 合并候选并选择最优
            candidates += beam  # 保留已完成的候选
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 去重并选择top-k
            seen = set()
            new_beam = []
            for cand in candidates:
                key = tuple(cand['tokens'])
                if key not in seen and len(new_beam) < beam_width:
                    seen.add(key)
                    new_beam.append(cand)
            beam = new_beam

        # 选择最佳序列
        best_candidate = max(beam, key=lambda x: x['score'])
        if best_candidate['tokens'][-1] != eos_token_id:
            best_candidate['tokens'].append(eos_token_id)
            
        return tokenizer.decode(best_candidate['tokens'], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description='神经机器翻译推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--tokenizer_dir', type=str, required=True, help='分词器目录')
    parser.add_argument('--input_text', type=str, required=True, help='要翻译的英文文本')
    parser.add_argument('--max_length', type=int, default=64, help='最大序列长度')
    parser.add_argument('--beam_width', type=int, default=5, help='束搜索宽度（仅束搜索模式有效）')
    parser.add_argument('--mode', choices=['greedy', 'beam'], default='greedy', help='解码模式')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model, tokenizer = load_model(
        args.model_path,
        args.tokenizer_dir,
        args.max_length,
        device
    )

    # 执行解码
    if args.mode == 'greedy':
        result = greedy_decode(
            model,
            tokenizer,
            args.input_text,
            args.max_length,
            device
        )
    else:
        result = beam_search_decode(
            model,
            tokenizer,
            args.input_text,
            args.max_length,
            device,
            args.beam_width
        )

    print(f"\n输入文本: {args.input_text}")
    print(f"翻译结果: {result}")


if __name__ == "__main__":
    main()
