"""Bucla de antrenare pe o epocă și evaluarea pe un subset (validare sau test), folosite de ``train.py`` și ``test.py``."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.utils import character_error_rate, ctc_greedy_decode, word_error_rate


def ctc_loss(criterion: nn.CTCLoss, log_probs: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor):
    """Aplică CTC: toate secvențele de ieșire au aceeași lungime T (imaginile au aceeași lățime după padding).
    Calculul se face în float32, chiar dacă rețeaua rulează în precizie mixtă (CTC este sensibil numeric)."""
    t, b = log_probs.shape[:2]
    input_lengths = torch.full((b,), t, dtype=torch.long)
    return criterion(log_probs.float(), targets, input_lengths, target_lengths)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, *, use_amp: bool = True,
                    grad_clip: float = 5.0, max_batches: int | None = None, desc: str = 'train') -> float:
    """O epocă de antrenare; întoarce media funcției de cost CTC pe batch-uri."""
    model.train()
    total, count = 0.0, 0
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)
    progress = tqdm(loader, total=n_batches, desc=desc, leave=False)
    for step, (images, targets, target_lengths, _) in enumerate(progress):
        if max_batches and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
            log_probs = model(images)
        loss = ctc_loss(criterion, log_probs, targets, target_lengths)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                      # gradienții reali, pentru tăiere
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # evită exploziile de gradient din LSTM
        scaler.step(optimizer)
        scaler.update()

        if math.isfinite(loss.item()):
            total += loss.item()
            count += 1
        progress.set_postfix(loss=f'{total / max(count, 1):.3f}')
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, charset, device, *, use_amp: bool = True, max_batches: int | None = None,
             desc: str = 'eval') -> dict:
    """Evaluează modelul: funcția de cost CTC, CER și WER (decodare greedy) și predicțiile pentru fiecare cuvânt."""
    model.eval()
    total, count = 0.0, 0
    references, predictions = [], []
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)
    for step, (images, targets, target_lengths, texts) in enumerate(tqdm(loader, total=n_batches, desc=desc, leave=False)):
        if max_batches and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
            log_probs = model(images)
        loss = ctc_loss(criterion, log_probs, targets, target_lengths)
        if math.isfinite(loss.item()):
            total += loss.item()
            count += 1
        references += texts
        predictions += ctc_greedy_decode(log_probs.float(), charset)
    return {
        'loss': total / max(count, 1),
        'cer': character_error_rate(references, predictions),
        'wer': word_error_rate(references, predictions),
        'references': references,
        'predictions': predictions,
    }
