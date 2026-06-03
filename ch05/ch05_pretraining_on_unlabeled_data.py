"""
ch05_pretraining_on_unlabeled_data.py

Importable helper module for Chapter 5.

This file contains ONLY the function definitions needed by
`train_model_simple` and its dependencies. There is no top-level
executable code (no data loading, no model building, no training,
no printing), so importing it is fast and silent:

    from ch05_pretraining_on_unlabeled_data import train_model_simple

Assumes the standard book layout, where ch04 is a sibling folder of
ch05, so that `generate_text_simple` can be imported from
`ch04_gpt_model`.
"""

import os
import sys

import torch

# Make the sibling ch04 folder importable regardless of the machine /
# working directory, so this file is portable (no hard-coded absolute path).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CH04 = os.path.join(_HERE, "..", "ch04")
if _CH04 not in sys.path:
    sys.path.append(_CH04)

from ch04_gpt_model import generate_text_simple


# --- Tokenisation helpers --------------------------------------------------

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# --- Loss helpers ----------------------------------------------------------

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# --- Evaluation + sampling -------------------------------------------------

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


# --- Main training loop ----------------------------------------------------

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


# --- Data setup builder ----------------------------------------------------

def get_data_loaders(gpt_config, data_path=None, train_ratio=0.9, batch_size=2):
    """Build and return the train/val DataLoaders plus a GPT-2 tokenizer.

    NOTE: this does not (and cannot) import live objects from another
    notebook. It imports the *code* needed to build them and then
    constructs fresh objects in the current kernel:
      - `create_dataloader_v1` is imported from the ch02 module,
      - the training text is read from `the-verdict.txt`,
      - the two loaders and the tokenizer are created here and returned.

    Parameters
    ----------
    gpt_config : dict
        The GPT config dict (e.g. GPT_CONFIG_124M); only "context_length"
        is used here.
    data_path : str, optional
        Path to the text file. Defaults to ``../ch02/the-verdict.txt``
        relative to this file.
    train_ratio : float
        Fraction of the text used for training (rest is validation).
    batch_size : int

    Returns
    -------
    (train_loader, val_loader, tokenizer)
    """
    import tiktoken

    # Make the sibling ch02 folder importable, and locate the text file.
    ch02 = os.path.join(_HERE, "..", "ch02")
    if ch02 not in sys.path:
        sys.path.append(ch02)
    from ch02_tokenization import create_dataloader_v1

    if data_path is None:
        data_path = os.path.join(ch02, "the-verdict.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    ctx = gpt_config["context_length"]
    train_loader = create_dataloader_v1(
        train_data, batch_size=batch_size, max_length=ctx, stride=ctx,
        drop_last=True, shuffle=True, num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=batch_size, max_length=ctx, stride=ctx,
        drop_last=False, shuffle=False, num_workers=0,
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    return train_loader, val_loader, tokenizer