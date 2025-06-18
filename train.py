# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from MyDataset import MyDataset, collate_fn
from functools import partial
from SpikingParrot import SpikingParrot
from MyTokenizer import load_tokenizer
import os
from tqdm import tqdm
import gc
import logging
import math
from BucketManager import BucketManager
import random
import numpy as np
from TryBatch import find_batch_size


CONFIG = {
    # dir
    "train_cache_dir": "data/cache/train",
    "valid_cache_dir": "data/cache/valid",
    "checkpoint_dir": "model/checkpoints",

    # data settings
    "num_workers": 20,
    "safty_factor": 0.75,

    # model config
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout": 0.03,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # optimizer and scheduler config
    "epochs": 12,
    "lr_eta_max": 5e-4,
    "lr_eta_min": 5e-6,
    "tf_eta_max": 1,
    "tf_eta_min": 0.5,
    "betas": (0.9, 0.99),
    "weight_decay": 1e-4,

    "label_smoothing": 0.03,

    # grad and precision
    "use_amp": True,
    "grad_clip": 1.0,

    # logging
    "log_file": "training.log",
}


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

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_checkpoint(model, optimizer, scaler, lr_scheduler, tf_scheduler, epoch, bucket_idx, config):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'bucket_idx': bucket_idx,
        'bucket_list': config["bucket_list"],
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'lr_scheduler_state': lr_scheduler.state_dict(),
        'tf_scheduler_state': tf_scheduler.state_dict(),
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }
    # if whole epoch trained, save extra checkpoint
    if bucket_idx == len(config["bucket_list"]):
        torch.save(model, os.path.join(config["checkpoint_dir"], f"model_{epoch:03d}.pt"))
        torch.save(checkpoint, os.path.join(config["checkpoint_dir"], f"checkpoint_{epoch:03d}.pt"))
    # save latest
    torch.save(model, os.path.join(config["checkpoint_dir"], "latest_model.pt"))
    torch.save(checkpoint, os.path.join(config["checkpoint_dir"], "latest_checkpoint.pt"))


def load_latest_checkpoint(model, optimizer, scaler, lr_scheduler, tf_scheduler, config):
    checkpoint_path = os.path.join(config["checkpoint_dir"], "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        return model, optimizer, scaler, lr_scheduler, tf_scheduler, 0, 0, config["bucket_list"]
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
    tf_scheduler.load_state_dict(checkpoint['tf_scheduler_state'])
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    
    if scaler and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])

    return (
        model, optimizer, scaler,
        lr_scheduler, tf_scheduler,
        checkpoint['epoch'],
        checkpoint['bucket_idx'],
        checkpoint["bucket_list"]
    )


def validate(model, config):
    dataset = MyDataset(
        cache_path="data/cache/valid",
        src_range=(0, 128),
        tgt_range=(0, 128),
        shard_idx=0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=partial(
            collate_fn,
            pad_token_id=config["pad_token_id"],
            src_max_length=128,
            tgt_max_length=128,
        )
    )
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid_tokens = 0
    total_samples = 0
    
    with torch.no_grad():
        loop = tqdm(
            dataloader,
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
            
            output = model(src, tgt, 1)
            loss = nn.CrossEntropyLoss(
                ignore_index=config["pad_token_id"],
                label_smoothing=config["label_smoothing"],
            )(
                output.reshape(-1, output.size(-1)),
                labels.contiguous().view(-1)
            )
            
            preds = torch.argmax(output, dim=-1)
            mask = (labels != config["pad_token_id"])
            correct = (preds[mask] == labels[mask]).sum()

            total_valid = mask.sum()
            total_loss += loss.item() * batch_size
            total_correct += correct.item()
            total_valid_tokens += total_valid.item()
            total_samples += batch_size
            
            loop.set_postfix(
                loss=f'{loss.item():.3f}',
                acc=f'{(correct.float() / total_valid).item() * 100:.3f}%' \
                    if total_valid > 0 else '0.000%',
            )
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    del dataset, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return avg_loss, avg_acc


def train(config):
    logger = setup_logger(config["log_file"])

    # init tokenizer
    en_tokenizer = load_tokenizer("model/tokenizers/en")
    zh_tokenizer = load_tokenizer("model/tokenizers/zh")
    assert \
        zh_tokenizer.bos_token_id == en_tokenizer.bos_token_id and \
        zh_tokenizer.eos_token_id == en_tokenizer.eos_token_id and \
        zh_tokenizer.unk_token_id == en_tokenizer.unk_token_id and \
        zh_tokenizer.pad_token_id == en_tokenizer.pad_token_id and \
        zh_tokenizer.vocab_size == en_tokenizer.vocab_size

    config["bos_token_id"] = zh_tokenizer.bos_token_id
    config["eos_token_id"] = zh_tokenizer.eos_token_id
    config["unk_token_id"] = zh_tokenizer.unk_token_id
    config["pad_token_id"] = zh_tokenizer.pad_token_id
    config["vocab_size"] = zh_tokenizer.vocab_size

    # init bucket manager
    buckets = BucketManager(
        cache_path=config["train_cache_dir"],
        force_rebuild=False
    )

    # buckets.reset_batch_size()
    if not buckets.found_optimal():
        buckets.find_optimal_batch_size(partial(find_batch_size, config=config))

    config["bucket_list"] = buckets.get_info()
    config["T_max"] = buckets.get_total_iterations(safety_factor=config["safty_factor"])

    # init model
    model = SpikingParrot(
        padding_idx=config["pad_token_id"],
        vocab_size=config["vocab_size"],
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
        T_max=config["T_max"] * 2,
        eta_min=config["lr_eta_min"]
    )
    tf_scheduler = CosineAnnealingTFR(
        T_max=config["T_max"],
        eta_max=config["tf_eta_max"],
        eta_min=config["tf_eta_min"]
    )
    scaler = GradScaler(config["device"], enabled=config["use_amp"])

    # recover from checkpoint
    start_epoch = 0
    start_bucket_idx = 0

    if os.path.exists(os.path.join(config["checkpoint_dir"], "latest_checkpoint.pt")):
        model, optimizer, scaler, lr_scheduler, tf_scheduler, start_epoch, start_bucket_idx, config["bucket_list"] = load_latest_checkpoint(
            model, optimizer, scaler, lr_scheduler, tf_scheduler, config
        )

    if start_epoch >= config["epochs"]:
        logger.info("Training already completed.")
        return

    # main train loop
    for epoch in range(start_epoch, config["epochs"]):
        for bucket_idx in range(start_bucket_idx, len(config["bucket_list"])):
            
            bucket = config["bucket_list"][bucket_idx]
            
            logger.info(
                f"Processing bucket {bucket_idx + 1}/{len(config['bucket_list'])} " + \
                f"src: {str(bucket['src_range'])} tgt: {str(bucket['tgt_range'])} " + \
                f"shard: {str(bucket['shard_idx'] + 1)}/{str(bucket['num_shards'])}"
            )

            dataset = MyDataset(
                cache_path=config["train_cache_dir"],
                src_range=bucket["src_range"],
                tgt_range=bucket["tgt_range"],
                shard_idx=bucket["shard_idx"]
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=math.floor(bucket["suggested_batch_size"] * config["safty_factor"]),
                num_workers=config["num_workers"],
                collate_fn=partial(
                    collate_fn,
                    pad_token_id=config["pad_token_id"],
                    src_max_length=bucket["src_range"][1],
                    tgt_max_length=bucket["tgt_range"][1],
                ),
                shuffle=False,
            )
            
            # train loop
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_valid_tokens = 0
            total_samples = 0

            loop = tqdm(
                dataloader,
                unit="batch",
                colour="green",
                desc=f"Epoch {epoch + 1} | Bucket {bucket_idx + 1}",
                bar_format='{l_bar}{bar:32}{r_bar}',
                dynamic_ncols=True,
                leave=False,
            )
            
            for batch in loop:
                src = batch["src"].to(config["device"])
                tgt = batch["tgt"].to(config["device"])
                labels = batch["label"].to(config["device"])
                batch_size = src.size(0)
                
                # mixed precision
                with autocast(config["device"], enabled=config["use_amp"]):
                    if epoch % 3 == 0:
                        tfr = 1
                    else:
                        tfr = tf_scheduler.get_last_tfr()
                    output = model(src, tgt, tfr)
                    loss = nn.CrossEntropyLoss(
                        ignore_index=config["pad_token_id"],
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
                optimizer.zero_grad(set_to_none=True)

                # update scheduler after a batch
                lr_scheduler.step()
                if epoch % 3 != 0:
                    tf_scheduler.step()

                # preformance eval
                with torch.no_grad():
                    preds = torch.argmax(output, dim=-1)
                    mask = (labels != config["pad_token_id"])
                    correct = (preds[mask] == labels[mask]).sum()

                    total_valid = mask.sum()
                    total_loss += loss.item() * batch_size
                    total_correct += correct.item()
                    total_valid_tokens += total_valid.item()
                    total_samples += batch_size
                    
                    loop.set_postfix(
                        loss=f'{loss.item():.3f}',
                        acc=f'{(correct.float() / total_valid).item() * 100:.3f}%' \
                            if total_valid > 0 else '0.000%',
                        lr=f'{lr_scheduler.get_last_lr()[0]:.3e}',
                        tf=f'{tfr:.3e}'
                    )
            
            # log parquet info
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
            logger.info(
                f"Epoch: {epoch + 1} | Bucket: {bucket_idx + 1} | Loss: {avg_loss:.3f} | Acc: {avg_acc * 100:.3f}%"
            )

            # shuffle after a epoch
            if bucket_idx == len(config["bucket_list"]) - 1:
                random.shuffle(config["bucket_list"])

            # save checkpoint after a bucket
            save_checkpoint(
                model, optimizer, scaler, lr_scheduler, tf_scheduler,
                epoch, bucket_idx + 1, config
            )

            # clean up memory
            del dataset, dataloader
            gc.collect()
            torch.cuda.empty_cache()
            
            # validate after a bucket
            valid_loss, valid_acc = validate(model, config)
            logger.info(f"Validation | Loss: {valid_loss:.3f} | Acc: {valid_acc * 100:.3f}%")

        # reset index
        start_bucket_idx = 0


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train(CONFIG)
