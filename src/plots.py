"""Graficele rezultatelor: evoluția antrenării și a validării pe epoci, rezultatele pe test și exemple de predicții.

Graficele se salvează ca fișiere PNG (fără fereastră), deci funcționează și pe un server sau în Colab.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: dict, path, test_metrics: dict | None = None, best_epoch: int | None = None):
    """Trei grafice pe epoci: (1) loss CTC train vs. validare, (2) CER și WER pe validare, (3) rata de învățare.
    Dacă există rezultatele pe test, acestea apar ca linii orizontale punctate (o singură valoare, modelul final)."""
    epochs = history['epoch']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'o-', ms=3, label='train')
    ax.plot(epochs, history['val_loss'], 'o-', ms=3, label='validare')
    if test_metrics:
        ax.axhline(test_metrics['loss'], color='C2', ls='--', label=f"test ({test_metrics['loss']:.3f})")
    ax.set_title('Funcția de cost CTC')
    ax.set_xlabel('epoca')
    ax.set_ylabel('loss')

    ax = axes[1]
    ax.plot(epochs, 100 * np.array(history['val_cer']), 'o-', ms=3, color='C1', label='CER validare')
    ax.plot(epochs, 100 * np.array(history['val_wer']), 's-', ms=3, color='C4', label='WER validare')
    if test_metrics:
        ax.axhline(100 * test_metrics['cer'], color='C1', ls='--', label=f"CER test ({100 * test_metrics['cer']:.2f}%)")
        ax.axhline(100 * test_metrics['wer'], color='C4', ls='--', label=f"WER test ({100 * test_metrics['wer']:.2f}%)")
    if best_epoch is not None and best_epoch in epochs:
        i = epochs.index(best_epoch)
        ax.plot(best_epoch, 100 * history['val_cer'][i], '*', ms=14, color='k',
                label=f"cel mai bun model (epoca {best_epoch})")
    ax.set_title('Rata de eroare (decodare greedy)')
    ax.set_xlabel('epoca')
    ax.set_ylabel('%')

    ax = axes[2]
    ax.plot(epochs, history['lr'], 'o-', ms=3, color='C5')
    ax.set_yscale('log')
    ax.set_title('Rata de învățare')
    ax.set_xlabel('epoca')

    for ax in axes:
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_test_analysis(rows: list[dict], path):
    """Analiza rezultatelor pe test: CER în funcție de lungimea cuvântului și distribuția CER pe cuvânt.
    rows: listă de dicționare cu cheile ``text``, ``prediction``, ``cer``."""
    lengths = np.array([len(r['text']) for r in rows])
    cers = np.array([r['cer'] for r in rows])
    buckets = [(1, 1), (2, 3), (4, 5), (6, 7), (8, 10), (11, 100)]
    labels = ['1', '2-3', '4-5', '6-7', '8-10', '11+']
    values, counts = [], []
    for low, high in buckets:
        mask = (lengths >= low) & (lengths <= high)
        counts.append(int(mask.sum()))
        # CER pe grup = erorile totale / caracterele totale ale grupului
        values.append(100 * (cers[mask] * lengths[mask]).sum() / max(1, lengths[mask].sum()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bars = axes[0].bar(labels, values, color='C1')
    for bar, n in zip(bars, counts):
        axes[0].annotate(f'n={n}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', fontsize=8)
    axes[0].set_title('CER pe test în funcție de lungimea cuvântului')
    axes[0].set_xlabel('numărul de caractere')
    axes[0].set_ylabel('CER (%)')

    exact = 100 * (cers == 0).mean()
    axes[1].hist(np.clip(100 * cers, 0, 100), bins=20, range=(0, 100), color='C0')
    axes[1].set_yscale('log')
    axes[1].set_title(f'Distribuția CER pe cuvânt ({exact:.1f}% cuvinte recunoscute exact)')
    axes[1].set_xlabel('CER pe cuvânt (%)')
    axes[1].set_ylabel('număr de cuvinte (log)')
    for ax in axes:
        ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_examples(images: list, texts: list[str], predictions: list[str], path, ncols: int = 4, title: str = ''):
    """Grilă de imagini de test cu eticheta reală și predicția (verde = corect, roșu = greșit)."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 1.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, img, text, pred in zip(axes.flat, images, texts, predictions):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'real: "{text}"\nprezis: "{pred}"', fontsize=8, color='green' if pred == text else 'red')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
