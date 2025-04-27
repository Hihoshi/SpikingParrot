# inference.py
import torch
from model import SpikingParrot
from mytokenizer import load_tokenizer

def load_model(checkpoint_path, config):
    model = SpikingParrot(
        padding_idx=config['padding_idx'],
        embedding_dim=config['embedding_dim'],
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional']
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    return model

def main():
    en_tokenizer = load_tokenizer("model/tokenizers", "en")
    zh_tokenizer = load_tokenizer("model/tokenizers", "zh")

    checkpoint = torch.load("model/checkpoints/epoch_005.pt", map_location='cpu')
    config = checkpoint['config']
    config['padding_idx'] = zh_tokenizer.pad_token_id
    config['vocab_size'] = zh_tokenizer.vocab_size

    model = load_model("model/checkpoints/epoch_005.pt", config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    while True:
        src_text = input("输入英文句子 (输入 'q' 退出): ")
        if src_text.lower() == 'q':
            break
        
        src_encoded = en_tokenizer.encode(src_text)
        src_ids = torch.tensor([src_encoded], device=device)

        with torch.no_grad():
            greedy_output = model.greedy_decode(
                src_ids,
                bos_token_id=en_tokenizer.bos_token_id,
                eos_token_id=en_tokenizer.eos_token_id,
            )
            model.reset()
            beam_output = model.beam_search(
                src_ids,
                bos_token_id=en_tokenizer.bos_token_id,
                eos_token_id=en_tokenizer.eos_token_id,
            )
            model.reset()

        print("贪心解码:", zh_tokenizer.decode(greedy_output[0], skip_special_tokens=True))
        print("集束搜索:", zh_tokenizer.decode(beam_output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
