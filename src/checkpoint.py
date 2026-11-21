"""Salvarea și încărcarea progresului antrenării (checkpoint).

Un checkpoint conține tot ce este necesar pentru a relua antrenarea exact de unde a rămas:

| Cheie | Conținut |
| --- | --- |
| ``epoch`` | ultima epocă terminată (antrenarea se reia de la ``epoch + 1``) |
| ``model_state`` | ponderile rețelei |
| ``optimizer_state`` | starea optimizatorului AdamW (momentele de ordinul 1 și 2 pentru fiecare parametru, pasul curent) |
| ``scheduler_state`` | starea scheduler-ului ReduceLROnPlateau (cea mai bună valoare, epocile fără îmbunătățire, rata curentă) |
| ``scaler_state`` | starea GradScaler (factorul de scalare pentru antrenarea în precizie mixtă) |
| ``rng_state`` | starea generatoarelor aleatoare: Python, NumPy, PyTorch CPU/GPU și generatorul DataLoader-ului de train |
| ``best_val_cer``, ``best_epoch`` | cel mai bun CER pe validare de până acum și epoca în care a fost obținut |
| ``history`` | evoluția metricilor pe epoci (loss train/val, CER, WER, rata de învățare, durata) |
| ``model_config``, ``train_config`` | hiperparametrii modelului și ai antrenării, inclusiv seed-ul |
| ``charset``, ``normalization`` | alfabetul și media / deviația standard folosite la normalizare (necesare la testare) |
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import torch

from src.utils import get_rng_state, set_rng_state


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, epoch: int, best_val_cer: float, best_epoch: int,
                    history: dict, model_config: dict, train_config: dict, charset, normalization: dict,
                    train_generator: torch.Generator | None = None):
    """Salvează checkpoint-ul. Scrierea se face întâi într-un fișier temporar, apoi acesta este redenumit: dacă
    programul este întrerupt în timpul salvării, checkpoint-ul anterior rămâne intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'rng_state': get_rng_state(train_generator),
        'best_val_cer': best_val_cer,
        'best_epoch': best_epoch,
        'history': history,
        'model_config': model_config,
        'train_config': train_config,
        'charset': charset.chars,
        'normalization': normalization,
        'torch_version': torch.__version__,
        'saved_at': dt.datetime.now().isoformat(timespec='seconds'),
    }
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location='cpu') -> dict:
    """Încarcă un checkpoint. ``weights_only=False`` este necesar deoarece checkpoint-ul conține și obiecte Python
    (starea RNG, istoricul); încărcați doar checkpoint-uri create de voi."""
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_training_state(checkpoint: dict, *, model, optimizer, scheduler, scaler,
                           train_generator: torch.Generator | None = None) -> int:
    """Restaurează modelul, optimizatorul, scheduler-ul, scaler-ul și starea RNG; întoarce epoca de la care se reia."""
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    if scaler is not None and checkpoint.get('scaler_state'):
        scaler.load_state_dict(checkpoint['scaler_state'])
    set_rng_state(checkpoint['rng_state'], train_generator)
    return checkpoint['epoch'] + 1
