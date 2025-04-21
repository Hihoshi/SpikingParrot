import torch
import torch.nn.functional as F
from model import SpikingParrot
from mytokenizer import load_tokenizer


def pad_batch(sequences, pad_id):
    # sequences: list of lists of ids
    max_len = max(len(seq) for seq in sequences)
    padded = [F.pad(torch.tensor(seq, dtype=torch.long), (0, max_len - len(seq)), value=pad_id) for seq in sequences]
    return torch.stack(padded, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizers
    en_tokenizer = load_tokenizer("model/tokenizers", "en")
    zh_tokenizer = load_tokenizer("model/tokenizers", "zh")

    # load model
    model = SpikingParrot(
        bidirectional=True,
        vocab_size=zh_tokenizer.vocab_size,
        embedding_dim=2048,
        hidden_dim=512,
        num_layers=4,
    ).to(device)
    model.load_state_dict(torch.load("model/checkpoints/epoch_004.pt")["model_state"])
    model.eval()

    # sample sentences
    sentences = [
        "Hello, how are you?"
    ]

    # tokenize and pad
    src_seqs = [en_tokenizer(sent)['input_ids'] for sent in sentences]
    src_padded = pad_batch(src_seqs, en_tokenizer.pad_token_id).to(device)

    # greedy decode
    greedy_ids = model.greedy_decode(
        src_padded,
        bos_id=zh_tokenizer.bos_token_id,
        eos_id=zh_tokenizer.eos_token_id,
        max_len=32
    )
    # convert to text
    greedy_texts = [zh_tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in greedy_ids]
    print("Greedy results:")
    for txt in greedy_texts:
        print(txt)

    # beam search on first sentence only
    beam_ids = model.beam_search(
        src_padded[:1],
        bos_id=zh_tokenizer.bos_token_id,
        eos_id=zh_tokenizer.eos_token_id,
        max_len=32,
        beam_size=4
    )
    beam_text = zh_tokenizer.decode(beam_ids, skip_special_tokens=True)
    print("Beam search result:")
    print(beam_text)


if __name__ == "__main__":
    main()
