import torch
import torch.nn as nn
from tqdm import tqdm


def init_weights1(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def init_weights2(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def init_weights3(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def init_weights4(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model):
    print(
        f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )


def train(
    epoch, model, dataloader, n_batch, optimizer, criterion, clip, device, mode="native"
):
    model.train()
    train_loss = 0
    with tqdm(desc=f"Epoch:{epoch+1: 2d}", total=n_batch) as pbar:
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            if mode == "native":
                src, trg = data
                src = src.to(device)
                trg = trg.to(device)
                output = model(src, trg)
            elif mode == "packed":
                src, src_len, trg = data
                src = src.to(device)
                trg = trg.to(device)
                output = model(src, src_len, trg)
            elif mode == "cnn":
                src, trg = data
                src = src.to(device)
                trg = trg.to(device)
                # trg [trg_len, bs]
                output, _ = model(src, trg[:-1, :])
                output = output.permute(1, 0, 2)
            # output [trg_len-1, bs, output_dim]
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item()
            avg_loss = train_loss / (i + 1)
            pbar.set_postfix({"train_loss": avg_loss})
            pbar.update()
            if i == n_batch - 1:
                break


def evaluate(model, dataloader, n_batch, criterion, device, mode="native"):
    model.eval()
    eval_loss = 0
    with tqdm(total=n_batch) as pbar:
        for i, data in enumerate(dataloader):
            if mode == "native":
                src, trg = data
                src = src.to(device)
                trg = trg.to(device)
                output = model(src, trg)
            elif mode == "packed":
                src, src_len, trg = data
                src = src.to(device)
                trg = trg.to(device)
                output = model(src, src_len, trg)
            elif mode == "cnn":
                src, trg = data
                src = src.to(device)
                trg = trg.to(device)
                # trg [trg_len, bs]
                output, _ = model(src, trg[:-1, :])
                output = output.permute(1, 0, 2)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            eval_loss += loss.item()
            avg_loss = eval_loss / (i + 1)
            pbar.set_postfix({"eval_loss": avg_loss})
            pbar.update()
            if i == n_batch - 1:
                break
    return avg_loss
