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


TRAINING_CONFIG = {
    # parameter config
    "batch_size": 449,
    "src_max_length": 48,
    "tgt_max_length": 48,
    
    # model config
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_layers": 4,
    "bidirectional": True,
    "dropout": 0.01,
    
    # optimizer config
    "learning_rate": 1e-3,
    "betas": (0.9, 0.99),
    "weight_decay": 0.01,
    
    # train config
    "epochs": 8,
    "grad_clip": 1.0,
    "label_smoothing": 0.01,
    "num_workers": 8,
    
    # lr scheduler config
    "scheduler_eta_min": 1e-6,
    
    "use_amp": True,
    "mixed_precision": True,
    
    # checkpoint path
    "checkpoint_dir": "model/checkpoints"
}


def save_checkpoint(model, optimizer, epoch, loss, acc, scaler, scheduler, config):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'loss': loss,
            'acc': acc,
            'config': config,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'scheduler_state': scheduler.state_dict(),
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
    device = torch.device("cuda")

    # 初始化分词器
    zh_tokenizer = load_tokenizer("model/tokenizers/zh")

    # 初始化数据集
    dataset = MyDataset(
        "model/tokenizers/en",
        "model/tokenizers/zh",
        "data/corpus.json",
        src_max_length=config["src_max_length"],
        tgt_max_length=config["tgt_max_length"],
        num_workers=config["num_workers"],
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
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"],
    )

    # init the model
    model = SpikingParrot(
        padding_idx=zh_tokenizer.pad_token_id,
        vocab_size=zh_tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout=config["dropout"],
    ).to(device)

    # optimizer config
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

    # mixed precision training
    scaler = GradScaler(enabled=config["mixed_precision"])

    # loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=zh_tokenizer.pad_token_id,
        label_smoothing=config["label_smoothing"],
    )

    # retrain after break
    start_epoch = 0
    if os.path.exists(config["checkpoint_dir"]):
        model, optimizer, scaler, scheduler, start_epoch = load_latest_checkpoint(
            model, optimizer, scaler, scheduler, config
        )
    print(f"Start training from epoch {start_epoch + 1}")

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

            with autocast(enabled=config["use_amp"]):
                output = model(src, tgt)
                loss = loss_fn(
                    output.reshape(-1, output.size(-1)),
                    labels.contiguous().view(-1)
                )

            # grad norm and clip
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config["grad_clip"]
            )

            # param update
            scaler.step(optimizer)
            scaler.update()

            model.reset()
            optimizer.zero_grad()

            # performance eval
            with torch.no_grad():
                preds = torch.argmax(torch.log_softmax(output, dim=-1), dim=-1).squeeze(dim=1)
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

        avg_loss = total_loss / len(dataloader.dataset)
        avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.3f} | Acc: {avg_acc*100:.3f}%")

        # update learning rate after a epoch and save checkpoint
        scheduler.step()
        save_checkpoint(
            model, optimizer, epoch, avg_loss, avg_acc,
            scaler, scheduler, config
        )


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    train(TRAINING_CONFIG)
