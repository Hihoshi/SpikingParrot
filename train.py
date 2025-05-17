# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from mydataset import MyDataset, collate_fn
from functools import partial
from spikingparrot import SpikingParrot
from mytokenizer import load_tokenizer
import os
from tqdm import tqdm
import glob
import gc
import random
import logging
import math


CONFIG = {
    # dir
    "train_dir": "data/corpus/parquet/train",
    "valid_dir": "data/corpus/parquet/valid",
    "train_cache_dir": "data/cache/train",
    "valid_cache_dir": "data/cache/valid",
    "checkpoint_dir": "model/checkpoints",

    # dataloader settings
    "batch_size": 1024,
    "src_max_length": 48,
    "tgt_max_length": 48,
    "num_workers": 16,

    # model config
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout": 0.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # optimizer and scheduler config
    "epochs": 8,
    "lr_eta_max": 1e-3,
    "lr_eta_min": 1e-5,
    "tf_eta_max": 0.8,
    "tf_eta_min": 0.2,
    "lr_T_max": 4096,
    "tf_T_max": 2048,
    "betas": (0.9, 0.99),
    "weight_decay": 0.01,

    "label_smoothing": 0.1,

    # grad and precision
    "use_amp": True,
    "grad_clip": 1.0,

    # logging
    "log_file": "training.log",
}

train_parquets = sorted(glob.glob(os.path.join(CONFIG["train_dir"], "*.parquet")))
if not train_parquets:
    raise ValueError(f"No training files found in {CONFIG['train_dir']}")
CONFIG["train_parquets"] = train_parquets


class CosineAnnealingTFR:
    def __init__(self, T_max, eta_max=1.0, eta_min=0.0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.last_epoch = last_epoch
        
        # 初始化状态
        self.current_tfr = eta_max if last_epoch == -1 else self._compute_tfr(last_epoch)
        if last_epoch == -1:
            self.step(0)

    def _compute_tfr(self, epoch):
        cos = math.cos(math.pi * epoch / self.T_max)
        return self.eta_min + (self.eta_max - self.eta_min) * (1 + cos) / 2

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        
        self.current_tfr = self._compute_tfr(self.last_epoch)
        
        if self.verbose:
            print(f'Epoch {self.last_epoch:5d}: TFR adjusted to {self.current_tfr:.4f}')

    def get_last_tfr(self):
        return self.current_tfr

    def state_dict(self):
        return {
            'T_max': self.T_max,
            'eta_max': self.eta_max,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch,
            'verbose': self.verbose
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.current_tfr = self._compute_tfr(self.last_epoch)


def setup_logger(log_file):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # file handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def save_checkpoint(model, optimizer, epoch, parquet_idx, loss, acc, scaler, lr_scheduler, tf_scheduler, config):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'file_idx': parquet_idx,
        'loss': loss,
        'acc': acc,
        'config': config,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'lr_scheduler_state': lr_scheduler.state_dict(),
        'tf_scheduler_state': tf_scheduler.state_dict()
    }
    # if whole epoch trained, save extra checkpoint
    if parquet_idx == len(config["train_parquets"]):
        torch.save(model, os.path.join(config["checkpoint_dir"], f"model_{epoch:03d}.pt"))
        torch.save(checkpoint, os.path.join(config["checkpoint_dir"], f"checkpoint_{epoch:03d}.pt"))
    # save latest
    torch.save(model, os.path.join(config["checkpoint_dir"], "latest_model.pt"))
    torch.save(checkpoint, os.path.join(config["checkpoint_dir"], "latest_checkpoint.pt"))


def load_latest_checkpoint(model, optimizer, scaler, lr_scheduler, tf_scheduler, config):
    checkpoint_path = os.path.join(config["checkpoint_dir"], "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        return model, optimizer, scaler, lr_scheduler, tf_scheduler, 0, 0, None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    
    if scaler and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    if lr_scheduler and 'lr_scheduler_state' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
    if tf_scheduler and 'tf_scheduler_state' in checkpoint:
        tf_scheduler.load_state_dict(checkpoint['tf_scheduler_state'])

    return (
        model, optimizer, scaler,
        lr_scheduler, tf_scheduler,
        checkpoint['epoch'],
        checkpoint['file_idx']
    )


def validate(model, zh_tokenizer, config):
    valid_parquets = sorted(glob.glob(os.path.join(config["valid_dir"], "*.parquet")))
    if not valid_parquets:
        raise ValueError(f"No training files found in {config['valid_dir']}")
    
    valid_dataset = MyDataset(
        en_tokenizer_dir="model/tokenizers/en",
        zh_tokenizer_dir="model/tokenizers/zh",
        parquet_file=valid_parquets[0],  # there is only one validation file
        cache_dir=config["valid_cache_dir"],
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
            src = batch["src"].to(config["device"])
            tgt = batch["tgt"].to(config["device"])
            labels = batch["label"].to(config["device"])
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
    logger = setup_logger(config["log_file"])

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
        dropout=config["dropout"],
    ).to(config["device"])

    # init optimier and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_eta_max"],
        betas=config["betas"],
        weight_decay=config["weight_decay"]
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["lr_T_max"],
        eta_min=config["lr_eta_min"]
    )
    tf_scheduler = CosineAnnealingTFR(
        T_max=config["tf_T_max"],
        eta_max=config["tf_eta_max"],
        eta_min=config["tf_eta_min"]
    )
    scaler = GradScaler(enabled=config["use_amp"])
    
    # recover from checkpoint
    start_epoch = 0
    start_parquet_idx = 0
    if os.path.exists(os.path.join(config["checkpoint_dir"], "latest_checkpoint.pt")):
        model, optimizer, scaler, lr_scheduler, tf_scheduler, start_epoch, start_parquet_idx = load_latest_checkpoint(
            model, optimizer, scaler, lr_scheduler, tf_scheduler, config
        )
    if start_parquet_idx >= len(config["train_parquets"]):
        start_epoch += 1
        start_parquet_idx = 0
    if start_epoch >= config["epochs"]:
        logger.info("Training already completed.")
        return

    # main train loop
    for epoch in range(start_epoch, config["epochs"]):
        for parquet_idx in range(start_parquet_idx, len(config["train_parquets"])):
            parquet = config["train_parquets"][parquet_idx]
            logger.info(f"Processing {os.path.basename(parquet)} {parquet_idx+1}/{len(config['train_parquets'])}")
            
            # load parquet datset
            train_dataset = MyDataset(
                en_tokenizer_dir="model/tokenizers/en",
                zh_tokenizer_dir="model/tokenizers/zh",
                parquet_file=parquet,
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
                pin_memory=True,
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
                desc=f"Epoch {epoch + 1} | Segment {parquet_idx + 1}",
                bar_format='{l_bar}{bar:32}{r_bar}',
                dynamic_ncols=True,
                leave=False,
            )
            
            for batch in loop:
                optimizer.zero_grad(set_to_none=True)
                src = batch["src"].to(config["device"])
                tgt = batch["tgt"].to(config["device"])
                labels = batch["label"].to(config["device"])
                batch_size = src.size(0)
                
                # mixed precision
                with autocast(enabled=config["use_amp"]):
                    tfr = tf_scheduler.get_last_tfr()
                    output = model(src, tgt, tfr)
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
                # update scheduler after a batch
                lr_scheduler.step()
                tf_scheduler.step()
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
                        lr=f'{lr_scheduler.get_last_lr()[0]:.2e}',
                        tf=f'{tf_scheduler.get_last_tfr():.2e}'
                    )
            
            # log parquet info
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
            logger.info(
                f"Epoch: {epoch + 1} | Segment: {parquet_idx + 1} | Loss: {avg_loss:.3f} | Acc: {avg_acc*100:.3f}%"
            )

            # save checkpoint after a parquet
            save_checkpoint(
                model, optimizer, epoch, parquet_idx + 1,
                avg_loss, avg_acc, scaler, lr_scheduler, tf_scheduler, config
            )

            # clean up memory
            del train_dataset, train_dataloader
            gc.collect()
            torch.cuda.empty_cache()
            
            # validate after a parquet
            valid_loss, valid_acc = validate(model, zh_tokenizer, config)
            logger.info(f"Validation | Loss: {valid_loss:.3f} | Acc: {valid_acc*100:.3f}%")

        # reset index and reshuffle
        start_parquet_idx = 0
        if epoch != 0:
            random.shuffle(config["train_parquets"])


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train(CONFIG)
