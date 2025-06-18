# try_batch.py
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from SpikingParrot import SpikingParrot
import multiprocessing
import gc
import math
import random


def try_batch(config: dict, src_max: int, tgt_max: int, batch_size: int, max_batch_iter: int = 2):
    # init model, optimizer, scaler
    model = SpikingParrot(
        padding_idx=config["pad_token_id"],
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(config["device"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_eta_max"],
        betas=config["betas"],
        weight_decay=config["weight_decay"]
    )
    
    scaler = GradScaler(config["device"], enabled=config["use_amp"])

    # construct fake batch for testing
    srcs = torch.full((1, src_max), config["unk_token_id"], dtype=torch.long).expand(batch_size, -1).to(config["device"])
    tgts = torch.full((1, tgt_max), config["unk_token_id"], dtype=torch.long).expand(batch_size, -1).to(config["device"])
    labels = torch.full((1, tgt_max), config["unk_token_id"], dtype=torch.long).expand(batch_size, -1).to(config["device"])
    
    # training loop
    for i in range(max_batch_iter):
        with autocast(config["device"], enabled=config["use_amp"]):
            if i % 2 == 0:
                tfr = random.random()
            else:
                tfr = 1
            output = model(srcs, tgts, tfr)
            loss = nn.CrossEntropyLoss()(
                output.reshape(-1, output.size(-1)),
                labels.contiguous().view(-1)
            )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config["grad_clip"]
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def _worker(config, src_max, tgt_max, batch_size, max_batch_iter, result_queue):
    try:
        torch.cuda.reset_peak_memory_stats()
        try_batch(config, src_max, tgt_max, batch_size, max_batch_iter)
        peak_mem = torch.cuda.max_memory_allocated()
        result_queue.put(('success', peak_mem))
    except Exception as e:
        if 'out of memory' in str(e).lower() or 'alloc' in str(e).lower():
            peak_mem = torch.cuda.max_memory_allocated()
            result_queue.put(('oom', peak_mem))
        else:
            result_queue.put(('error', type(e).__name__, str(e)))
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_batch_size(
    src_max: int,
    tgt_max: int,
    config: dict,
    initial_batch_size: int = 16384,
    max_batch_iter: int = 2,
    timeout: int = 30
) -> int:
    low, high = 1, initial_batch_size
    best_size = 0
    max_attempts = math.ceil(math.log2(initial_batch_size))
    attempt = 0

    while low <= high and attempt < max_attempts:
        attempt += 1
        mid = (low + high) // 2
        
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_worker,
            args=(config, src_max, tgt_max, mid, max_batch_iter, result_queue)
        )
        
        try:
            torch.cuda.reset_peak_memory_stats()
            process.start()
            process.join(timeout=timeout)
            
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError(f"Batch size {mid} timed out after {timeout}s")
            
            if result_queue.empty():
                raise RuntimeError("Subprocess exited without returning results")

            status, data = result_queue.get()
            
            if status == 'success':
                current_peak = data / (1024 ** 3)
                print(f"Attempt {attempt:3d}: {mid:5d} | OK     (Peak Memory: {current_peak:5.2f}GB)")
                best_size = mid
                low = mid + 1
            elif status == 'oom':
                current_peak = data / (1024 ** 3)
                print(f"Attempt {attempt:3d}: {mid:5d} | Failed (OOM, Peak: {current_peak:5.2f}GB)")
                high = mid - 1
            elif status == 'error':
                exc_type, exc_msg = data
                print(f"Attempt {attempt:3d}: {mid:5d} | CRITICAL ERROR: {exc_type} - {exc_msg}")
                raise RuntimeError(f"Critical error at batch size {mid}: {exc_type} - {exc_msg}")

        except Exception as e:
            print(f"Attempt {attempt:3d}: {mid:5d} | Error: {str(e)}")
            high = mid - 1
        
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    return best_size
