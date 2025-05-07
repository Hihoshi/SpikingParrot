# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from mydataset import MyDataset, collate_fn
from functools import partial
from model import SpikingParrot
from mytokenizer import load_tokenizer
import os
from tqdm import tqdm
import glob
import gc
import random


TRAINING_CONFIG = {
    # dir
    "train_dir": "data/corpus/parquet/train_shuffled",
    "valid_dir": "data/corpus/parquet/valid/validation-00000-of-00001.parquet",
    "train_cache_dir": "data/cache/train",
    "valid_cache_dir": "data/cache/valid",
    "checkpoint_dir": "model/checkpoints",

    # dataloader settings
    "batch_size": 601,
    "src_max_length": 48,
    "tgt_max_length": 48,
    "num_workers": 16,

    # model config
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_layers": 4,
    "bidirectional": True,
    "dropout": 0.03,

    # train config
    "learning_rate": 1e-3,
    "betas": (0.9, 0.99),
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "label_smoothing": 0.03,
    "teacher_forcing_ratio": [0.8, 0.5, 0.2, 0, 0],

    "epochs": 5,
    "scheduler_warmup_epochs": 2,
    "scheduler_eta_min": 1e-5,

    # mixed precision
    "use_amp": True
}


def save_checkpoint(model, optimizer, epoch, file_idx, parquet_files, loss, acc, scaler, scheduler, config):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'file_idx': file_idx,
        'parquet_files': parquet_files,
        'loss': loss,
        'acc': acc,
        'config': config,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    # if whole epoch trained, save extra checkpoint
    if file_idx == len(parquet_files) - 1:
        torch.save(model, os.path.join(config["checkpoint_dir"], f"model_{epoch:03d}.pt"))
        torch.save(checkpoint, os.path.join(config["checkpoint_dir"], f"checkpoint_{epoch:03d}.pt"))
    # save latest
    torch.save(checkpoint, os.path.join(config["checkpoint_dir"], "latest.pt"))


def load_latest_checkpoint(model, optimizer, scaler, scheduler, config):
    checkpoint_path = os.path.join(config["checkpoint_dir"], "latest.pt")
    if not os.path.exists(checkpoint_path):
        return model, optimizer, scaler, scheduler, 0, -1, None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    
    if scaler and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    return (
        model, optimizer, scaler, scheduler,
        checkpoint['epoch'],
        checkpoint['file_idx'],
        checkpoint['parquet_files']
    )


def create_scheduler(optimizer, config):
    warmup_epochs = config["scheduler_warmup_epochs"]
    total_epochs = config["epochs"]
    
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=config["scheduler_eta_min"]
    )
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )


def validate(model, valid_dir, valid_cache_dir, zh_tokenizer, config, device):
    valid_dataset = MyDataset(
        en_tokenizer_dir="model/tokenizers/en",
        zh_tokenizer_dir="model/tokenizers/zh",
        parquet_file=valid_dir,
        cache_dir=valid_cache_dir,
        src_max_length=config["src_max_length"],
        tgt_max_length=config["tgt_max_length"],
        num_workers=config["num_workers"],
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=partial(
            collate_fn,
            pad_token_id=zh_tokenizer.pad_token_id,
            src_max_length=config["src_max_length"],
            tgt_max_length=config["tgt_max_length"],
        )
    )
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid_tokens = 0
    total_samples = 0
    
    with torch.no_grad():
        loop = tqdm(
            valid_dataloader,
            unit="batch",
            colour="blue",
            desc="Validating",
            bar_format='{l_bar}{bar:32}{r_bar}',
            dynamic_ncols=True,
            leave=False
        )
        for batch in loop:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            labels = batch["label"].to(device)
            batch_size = src.size(0)
            
            output = model(src, tgt)
            loss = nn.CrossEntropyLoss(
                ignore_index=zh_tokenizer.pad_token_id,
                label_smoothing=config["label_smoothing"],
            )(
                output.reshape(-1, output.size(-1)),
                labels.contiguous().view(-1)
            )
            
            preds = torch.argmax(output, dim=-1)
            mask = (labels != zh_tokenizer.pad_token_id)
            correct = (preds[mask] == labels[mask]).sum()

            total_valid = mask.sum()
            total_loss += loss.item() * batch_size
            total_correct += correct.item()
            total_valid_tokens += total_valid.item()
            total_samples += batch_size
            
            loop.set_postfix(
                loss=f'{loss.item():.3f}',
                acc=f'{(correct.float()/total_valid).item()*100:.3f}%' \
                    if total_valid > 0 else '0.000%',
            )
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    del valid_dataset, valid_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return avg_loss, avg_acc


def train(config):
    device = torch.device("cuda")
    
    # init tokenizer
    en_tokenizer = load_tokenizer("model/tokenizers/en")
    zh_tokenizer = load_tokenizer("model/tokenizers/zh")
    assert zh_tokenizer.vocab_size == en_tokenizer.vocab_size, "tokenizers' vocab size is not consistent"
    
    # init model
    model = SpikingParrot(
        padding_idx=zh_tokenizer.pad_token_id,
        vocab_size=zh_tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout=config["dropout"],
    ).to(device)
    
    # init optimier and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        weight_decay=config["weight_decay"]
    )
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler(enabled=config["use_amp"])
    
    # load checkpoint
    start_epoch = 0
    start_file_idx = -1
    parquet_files = []
    if os.path.exists(os.path.join(config["checkpoint_dir"], "latest.pt")):
        model, optimizer, scaler, scheduler, start_epoch, start_file_idx, parquet_files = load_latest_checkpoint(
            model, optimizer, scaler, scheduler, config
        )
    
    # main train loop
    for epoch in range(start_epoch, config["epochs"]):
        # prepare parquets
        if not parquet_files:  # if new epoch
            parquet_files = sorted(glob.glob(os.path.join(config["train_dir"], "*.parquet")))
            if not parquet_files:
                raise ValueError(f"No training files found in {config['train_dir']}")
            random.shuffle(parquet_files)
        
        # recover from checkpoint
        start_idx = start_file_idx + 1 if epoch == start_epoch else 0
        if start_idx >= len(parquet_files):
            continue
        # train every parquet
        for file_idx in range(start_idx, len(parquet_files)):
            parquet_file = parquet_files[file_idx]
            print(f"\nProcessing {os.path.basename(parquet_file)} [{file_idx+1}/{len(parquet_files)}]")
            
            # load parquet datset
            train_dataset = MyDataset(
                en_tokenizer_dir="model/tokenizers/en",
                zh_tokenizer_dir="model/tokenizers/zh",
                parquet_file=parquet_file,
                cache_dir=config["train_cache_dir"],
                src_max_length=config["src_max_length"],
                tgt_max_length=config["tgt_max_length"],
                num_workers=config["num_workers"],
            )
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                collate_fn=partial(
                    collate_fn,
                    pad_token_id=zh_tokenizer.pad_token_id,
                    src_max_length=config["src_max_length"],
                    tgt_max_length=config["tgt_max_length"],
                )
            )
            
            # train loop
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_valid_tokens = 0
            total_samples = 0
            loop = tqdm(
                train_dataloader,
                unit="batch",
                colour="green",
                desc=f"Epoch {epoch + 1} | Segment {file_idx + 1}: ",
                bar_format='{l_bar}{bar:32}{r_bar}',
                dynamic_ncols=True,
                leave=False,
            )
            
            for batch in loop:
                optimizer.zero_grad(set_to_none=True)
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                labels = batch["label"].to(device)
                batch_size = src.size(0)
                
                # mixed precision
                with autocast(enabled=config["use_amp"]):
                    output = model(src, tgt, config["teacher_forcing_ratio"][epoch])
                    loss = nn.CrossEntropyLoss(
                        ignore_index=zh_tokenizer.pad_token_id,
                        label_smoothing=config["label_smoothing"]
                    )(
                        output.reshape(-1, output.size(-1)),
                        labels.contiguous().view(-1)
                    )
                
                # grad
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
                
                # preformance eval
                with torch.no_grad():
                    preds = torch.argmax(output, dim=-1)
                    mask = (labels != zh_tokenizer.pad_token_id)
                    correct = (preds[mask] == labels[mask]).sum()

                    total_valid = mask.sum()
                    total_loss += loss.item() * batch_size
                    total_correct += correct.item()
                    total_valid_tokens += total_valid.item()
                    total_samples += batch_size
                    
                    loop.set_postfix(
                        loss=f'{loss.item():.3f}',
                        acc=f'{(correct.float()/total_valid).item()*100:.3f}%' \
                            if total_valid > 0 else '0.000%',
                    )
            
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
            print(f"Epoch: {epoch + 1} | Segment {file_idx + 1} | Loss: {avg_loss:.3f} | Acc: {avg_acc*100:.2f}%")
            
            # save checkpoint after finishing a parquet
            save_checkpoint(
                model, optimizer, epoch, file_idx, parquet_files,
                avg_loss, avg_acc, scaler, scheduler, config
            )
            
            # clean up memory
            del train_dataset, train_dataloader
            gc.collect()
            torch.cuda.empty_cache()
            
            # validate after finishing a parquet
            valid_loss, valid_acc = validate(
                model, config["valid_dir"], config["valid_cache_dir"],
                zh_tokenizer, config, device
            )
            print(f"Validation | Loss: {valid_loss:.3f} | Acc: {valid_acc*100:.2f}%")
        
        # update after a epoch
        scheduler.step()
        parquet_files = []
        start_file_idx = -1


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train(TRAINING_CONFIG)
