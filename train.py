import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from mydataset import MyDataset, collate_fn
from functools import partial
from model import SpikingParrot
from mytokenizer import load_tokenizer
import os
from tqdm import tqdm

# 集中化的配置参数
TRAINING_CONFIG = {
    # 数据参数
    "batch_size": 797,
    "num_workers": 8,
    "src_max_length": 48,
    "tgt_max_length": 32,
    "seq_max_length": 64,
    
    # 模型架构参数
    "bidirectional": True,
    "embedding_dim": 2048,
    "hidden_dim": 512,
    "num_layers": 4,
    
    # 优化器参数
    "learning_rate": 1e-3,
    "betas": (0.9, 0.99),
    "weight_decay": 0.01,
    
    # 训练参数
    "epochs": 5,
    "grad_clip": 1.0,
    "label_smoothing": 0.03,
    
    # 学习率调度器
    "scheduler_eta_min": 1e-5,
    
    # 设备与精度
    "use_amp": True,
    "mixed_precision": True,
    
    # 检查点配置
    "checkpoint_dir": "model/checkpoints"
}


def save_checkpoint(model, optimizer, epoch, loss, acc, scaler, scheduler, config):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
            'scaler_state': scaler.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'config': config  # 保存配置信息
        },
        os.path.join(config["checkpoint_dir"], f"epoch_{epoch:03d}.pt")
    )


def load_latest_checkpoint(model, optimizer, scaler, scheduler, config):
    checkpoint_dir = config["checkpoint_dir"]
    if not os.path.exists(checkpoint_dir):
        return model, optimizer, scaler, scheduler, 0
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")]
    if not checkpoint_files:
        return model, optimizer, scaler, scheduler, 0
    
    latest_file = sorted(checkpoint_files)[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, latest_file))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    if scaler and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return model, optimizer, scaler, scheduler, checkpoint['epoch']


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化分词器
    en_tokenizer = load_tokenizer("model/tokenizers", "en")
    zh_tokenizer = load_tokenizer("model/tokenizers", "zh")

    # 初始化数据集
    dataset = MyDataset(
        en_tokenizer,
        zh_tokenizer,
        "data/corpus.json",
        src_max_length=config["src_max_length"],
        tgt_max_length=config["tgt_max_length"],
        seq_max_length=config["seq_max_length"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=partial(
            collate_fn,
            pad_token_id=zh_tokenizer.pad_token_id,
            src_max_length=config["src_max_length"],
            tgt_max_length=config["tgt_max_length"],
        ),
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    # 模型初始化
    model = SpikingParrot(
        bidirectional=config["bidirectional"],
        vocab_size=zh_tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["scheduler_eta_min"]
    )

    # 混合精度训练
    scaler = GradScaler(enabled=config["mixed_precision"])

    # 断点续训
    start_epoch = 0
    if os.path.exists(config["checkpoint_dir"]):
        model, optimizer, scaler, scheduler, start_epoch = load_latest_checkpoint(
            model, optimizer, scaler, scheduler, config
        )
    print(f"Start training from epoch {start_epoch + 1}")

    # 损失函数
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=zh_tokenizer.pad_token_id,
        label_smoothing=config["label_smoothing"],
    )

    for epoch in range(start_epoch, config["epochs"]):
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
            
            # 自动混合精度上下文
            with autocast(enabled=config["use_amp"]):
                output = model(src, tgt)
                loss = loss_fn(
                    output.reshape(-1, output.size(-1)),
                    labels.contiguous().view(-1)
                )
            
            # 梯度缩放和裁剪
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config["grad_clip"]
            )
            
            # 参数更新
            scaler.step(optimizer)
            scaler.update()
            model.reset()  # 重置神经元状态

            # 准确率计算
            with torch.no_grad():
                preds = torch.argmax(output, dim=-1).squeeze(dim=1)
                mask = (labels != zh_tokenizer.pad_token_id)
                correct = (preds[mask] == labels[mask]).sum()
                total_valid = mask.sum()
                
                batch_acc = correct.float() / total_valid if total_valid > 0 else 0.0
                total_loss += loss.item() * config["batch_size"]
                total_correct += correct.item()
                total_valid_tokens += total_valid.item()

            loop.set_postfix(
                loss=f'{loss.item():.3f}',
                acc=f'{batch_acc*100:.3f}%' if total_valid > 0 else '0.000%',
            )

        # 计算平均指标
        avg_loss = total_loss / len(dataloader.dataset)
        avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.3f} | Acc: {avg_acc*100:.3f}%")
        
        # 更新学习率和保存检查点
        scheduler.step()
        save_checkpoint(
            model, optimizer, epoch, avg_loss, avg_acc,
            scaler, scheduler, config
        )


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    train(TRAINING_CONFIG)
