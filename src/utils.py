"""Funcții auxiliare: reproductibilitate (seed și starea generatoarelor aleatoare), decodarea ieșirii CTC și metricile
CER / WER."""

from __future__ import annotations

import os
import random

import jiwer
import numpy as np
import torch


# ---------------------------------------------------------------------------------------------------------------------
# Reproductibilitate
# ---------------------------------------------------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False):
    """Fixează seed-ul pentru Python, NumPy și PyTorch (CPU și GPU).

    Cu ``deterministic=True``, cuDNN folosește doar algoritmi determiniști (rezultate identice la fiecare rulare, dar
    antrenare mai lentă); altfel ``cudnn.benchmark`` alege cei mai rapizi algoritmi pentru dimensiunea fixă a imaginilor.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_rng_state(train_generator: torch.Generator | None = None) -> dict:
    """Starea tuturor generatoarelor de numere aleatoare, pentru salvarea în checkpoint.

    ``train_generator`` este generatorul DataLoader-ului de train: de el depinde ordinea (amestecarea) batch-urilor,
    deci salvarea lui permite reluarea antrenării cu aceeași ordine a datelor ca într-o rulare neîntreruptă.
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if train_generator is not None:
        state['train_loader'] = train_generator.get_state()
    return state


def set_rng_state(state: dict, train_generator: torch.Generator | None = None):
    """Restaurează stările salvate de ``get_rng_state``."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
        except (RuntimeError, IndexError):
            print('Atenție: starea RNG CUDA nu a putut fi restaurată (alt număr de GPU-uri); se continuă fără ea.')
    if train_generator is not None and 'train_loader' in state:
        train_generator.set_state(state['train_loader'].cpu())


# ---------------------------------------------------------------------------------------------------------------------
# Decodarea CTC
# ---------------------------------------------------------------------------------------------------------------------

def ctc_greedy_decode(log_probs: torch.Tensor, charset) -> list[str]:
    """Decodare greedy (*best path*): la fiecare pas de timp se alege clasa cea mai probabilă, apoi se aplică regulile CTC:
    caracterele repetate consecutiv se unesc într-unul singur, iar simbolurile blank (indexul 0) se elimină.

    Exemplu: ``-hh-e-ll-ll-oo`` -> ``hello`` (``-`` = blank; blank-ul dintre cei doi ``l`` permite litere duble).

    log_probs: [T, B, num_classes] -> listă de B texte.
    """
    best = log_probs.argmax(dim=2).transpose(0, 1).cpu().tolist()  # [B, T]
    texts = []
    for sequence in best:
        indices, previous = [], 0
        for k in sequence:
            if k != previous and k != 0:
                indices.append(k)
            previous = k
        texts.append(charset.decode(indices))
    return texts


# ---------------------------------------------------------------------------------------------------------------------
# Metrici (calculate cu biblioteca jiwer)
# ---------------------------------------------------------------------------------------------------------------------

def character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """CER = (substituții + ștergeri + inserări) la nivel de caracter / numărul total de caractere din referințe.
    Se calculează pe întreg setul (nu ca medie a CER-urilor pe cuvânt), deci cuvintele lungi cântăresc mai mult."""
    return float(jiwer.cer(references, hypotheses))


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """WER = (substituții + ștergeri + inserări) la nivel de cuvânt / numărul total de cuvinte din referințe.
    Pentru cuvinte izolate (cazul IAM words) este practic fracțiunea de cuvinte recunoscute greșit."""
    return float(jiwer.wer(references, hypotheses))
