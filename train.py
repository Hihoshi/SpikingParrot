import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mydataset import MyDataset, collate_fn
from functools import partial
from model import SpikingParrot
from mytokenizer import load_tokenizer
import os
from tqdm import tqdm


def save_checkpoint(model, optimizer, epoch, loss, acc):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'loss': loss,
            'acc': acc
        },
        f"checkpoints/epoch_{epoch:03d}.pt"
    )


# 修改检查点加载函数（添加scaler处理）
def load_latest_checkpoint(model, optimizer):
    checkpoint_files = [f for f in os.listdir("checkpoints") if f.startswith("epoch_")]
    if not checkpoint_files:
        return model, optimizer, 0
    
    latest_file = sorted(checkpoint_files)[-1]
    checkpoint = torch.load(os.path.join("checkpoints", latest_file))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    return model, optimizer, checkpoint['epoch']


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化分词器
    tokenizer = load_tokenizer("model/tokenizers")

    # 初始化数据集
    dataset = MyDataset(
        tokenizer,
        "data/corpus.json",
        src_max_length=48,
        tgt_max_length=32,
        seq_max_length=64,
    )

    batch_size = 128
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_fn,
            pad_token_id=tokenizer.pad_token_id,
            src_max_length=48,
            tgt_max_length=32,
        ),
        num_workers=2,
        shuffle=True,
        drop_last=True,
    )

    # 模型初始化
    model = SpikingParrot(
        bidirectional=True,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=2048,
        hidden_dim=256,
        num_layers=2,
    ).to(device)

    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.99)
    )

    # 断点续训
    start_epoch = 0
    if os.path.exists("checkpoints"):
        model, optimizer, start_epoch = load_latest_checkpoint(model, optimizer)
    print(f"Start training from epoch {start_epoch + 1}")

    # 损失函数
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1,
    )

    for epoch in range(start_epoch, 5):
        model.train()
        total_loss = 0
        total_correct = 0
        total_valid_tokens = 0

        loop = tqdm(
            dataloader,
            unit="batch",
            colour="green",
            bar_format='{l_bar}{bar:32}{r_bar}',
            dynamic_ncols=True
        )

        for batch in loop:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(src, tgt)  # [batch, tgt_len, vocab_size]
            
            # 计算损失（维度调整）
            loss = loss_fn(
                output.reshape(-1, output.size(-1)),  # [batch*tgt_len, vocab]
                labels.contiguous().view(-1)          # [batch*tgt_len]
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 重置神经元状态
            model.reset()

            # 准确率计算
            with torch.no_grad():
                # 获取预测结果 [batch, tgt_len]
                preds = torch.argmax(output, dim=-1).squeeze(dim=1)
                
                # 创建mask [batch, tgt_len]
                mask = labels != tokenizer.pad_token_id
                
                # 统计有效预测
                correct = (preds[mask] == labels[mask]).sum()
                total_valid = mask.sum()
                
                batch_acc = correct.float() / total_valid if total_valid > 0 else 0.0
                
                # 累积统计量
                total_loss += loss.item() * batch_size  # 按样本数加权
                total_correct += correct.item()
                total_valid_tokens += total_valid.item()

            loop.set_postfix(
                loss=f'{loss.item():.3f}',
                acc=f'{batch_acc*100:.3f}%' if total_valid > 0 else '0.000%',
            )
        # 修改epoch统计部分
        avg_loss = total_loss / len(dataloader.dataset)  # 使用总样本数归一化
        avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.3f} | Acc: {avg_acc*100:.3f}%")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    train()
