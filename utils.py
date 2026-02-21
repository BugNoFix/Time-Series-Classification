import os
import copy
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter


# Data Loader

def make_loader(
    dataset,
    batch_size,
    shuffle=True,
    drop_last=False,
    nvidia_gpu=False,
    balanced_sampling=False,
    num_classes=3
):
    if balanced_sampling:
        labels = np.array([y for _, y in dataset])
        class_counts = np.bincount(labels, minlength=num_classes).astype(float)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not balanced_sampling,
        drop_last=drop_last,
        sampler=sampler,
        pin_memory=nvidia_gpu and torch.cuda.is_available(),
        pin_memory_device="cuda" if (nvidia_gpu and torch.cuda.is_available()) else "",
    )


# Input Device Helper

def move_inputs_to_device(inputs, device):
    """
    Moves batched inputs to device.
    Supports:
      - single tensor: x
      - tuple/list: (cont, survey, body, time)
    """
    if isinstance(inputs, (list, tuple)):
        return [x.to(device) for x in inputs]
    return inputs.to(device)


# Training

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    l1_lambda=0.0,
    l2_lambda=0.0,
    max_grad_norm=1.0
):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for inputs, targets in train_loader:
        inputs = move_inputs_to_device(inputs, device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            if isinstance(inputs, (list, tuple)):
                logits = model(*inputs)
            else:
                logits = model(inputs)

            loss = criterion(logits, targets)

            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            if l2_lambda > 0:
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm

        scaler.scale(loss).backward()

        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * targets.size(0)
        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_f1 = f1_score(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        average='weighted'
    )

    return epoch_loss, epoch_f1


# Validation

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = move_inputs_to_device(inputs, device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                if isinstance(inputs, (list, tuple)):
                    logits = model(*inputs)
                else:
                    logits = model(inputs)
                loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_f1 = f1_score(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        average='weighted'
    )

    return epoch_loss, epoch_f1

# Fit

def fit(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    optimizer,
    scaler,
    device,
    l1_lambda=0.0,
    l2_lambda=0.0,
    patience=0,
    evaluation_metric="val_f1",
    mode='max',
    restore_best_weights=True,
    verbose=10,
    experiment_name="",
    max_grad_norm=1.0
):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': []
    }

    if patience > 0:
        patience_counter = 0
        best_metric = float('-inf') if mode == 'max' else float('inf')
        best_epoch = 0

    print(f"Training {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            max_grad_norm=max_grad_norm
        )

        val_loss, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)


        if verbose > 0 and (epoch % verbose == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: Loss={train_loss:.4f}, F1={train_f1:.4f} | "
                f"Val: Loss={val_loss:.4f}, F1={val_f1:.4f}"
            )

        if patience > 0:
            current_metric = history[evaluation_metric][-1]
            improved = (current_metric > best_metric) if mode == 'max' else (current_metric < best_metric)

            if improved:
                best_metric = current_metric
                best_epoch = epoch
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), os.path.join("models", f"{experiment_name}_model.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    if restore_best_weights and patience > 0:
        model_path = os.path.join("models", f"{experiment_name}_model.pt")
        model.load_state_dict(torch.load(model_path))
        print(f"Best model restored from epoch {best_epoch} with {evaluation_metric}={best_metric:.4f}")

    if patience == 0:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("models", f"{experiment_name}_model.pt"))

    return model, history

def build_tcn_input(cont, survey, body, device):
    cont = cont.to(device)
    survey = survey.to(device)
    body = body.to(device)

    x = torch.cat([cont, survey.float(), body.float()], dim=-1)
    x = x.permute(0, 2, 1).contiguous()
    return x


def validate_one_epoch_tcn(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (cont, survey, body), yb in val_loader:
            yb = yb.to(device)

            x = build_tcn_input(cont, survey, body, device)
            logits = model(x)

            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / len(val_loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    micro_f1 = f1_score(all_targets, all_preds, average="micro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")

    return avg_loss, macro_f1, micro_f1, weighted_f1



def train_one_epoch_tcn(model, train_loader, optimizer, scaler, criterion, device, grad_clip=None):
    model.train()
    running_loss = 0.0

    for (cont, survey, body), yb in train_loader:
        yb = yb.to(device)
        x = build_tcn_input(cont, survey, body, device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, yb)

        if scaler:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * yb.size(0)

    return running_loss / len(train_loader.dataset)

