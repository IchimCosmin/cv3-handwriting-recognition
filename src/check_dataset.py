"""Verifică încărcarea setului de date: împărțirea train/val/test, alfabetul, forma batch-urilor și transformările.

Rulare (din folderul proiectului):
    python -m src.check_dataset --data_dir data

Afișează statisticile fiecărui subset și salvează în ``outputs/`` o imagine cu exemple din train (cu augmentări)
și din val (fără augmentări), împreună cu etichetele lor.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # salvăm figura în fișier, fără fereastră
import matplotlib.pyplot as plt
import torch

from src.dataset import get_dataloaders, label_stats


def show_batch(ax_row, images, texts, mean, std, title):
    for k, ax in enumerate(ax_row):
        img = images[k, 0] * std + mean  # inversăm normalizarea pentru afișare
        ax.imshow(img.clamp(0, 1).numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{title}: "{texts[k]}"', fontsize=9)
        ax.axis('off')


def main():
    parser = argparse.ArgumentParser(description='Verificarea Dataset-ului și a DataLoader-ului IAM')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--out_dir', default='outputs')
    args = parser.parse_args()

    loaders, charset, info = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    print('=== Împărțirea pe autori ===')
    for name in ['train', 'val', 'test']:
        print(f"{name:5s}: {info['n_words'][name]:6d} cuvinte, {info['n_writers'][name]:3d} autori"
              f" (eliminate pentru caractere necunoscute: {info['n_dropped_unknown_chars'][name]})")
    total = sum(info['n_words'].values())
    print('proporții (cuvinte):', {k: round(v / total, 3) for k, v in info['n_words'].items()})

    # niciun autor nu apare în două subseturi
    writers = {k: {s.writer for s in loaders[k].dataset.samples} for k in loaders}
    assert not (writers['train'] & writers['val']) and not (writers['train'] & writers['test']) \
        and not (writers['val'] & writers['test']), 'Același autor apare în mai multe subseturi!'
    print('Autorii sunt disjuncți între train / val / test: OK')

    print(f"\nAlfabet: {len(charset.chars)} caractere (+1 blank pentru CTC): {''.join(charset.chars)}")
    print('Normalizare (calculată pe train): mean = {:.4f}, std = {:.4f}'.format(info['mean'], info['std']))
    print('Etichete train:', label_stats(loaders['train'].dataset.samples))

    print('\n=== Un batch din fiecare DataLoader ===')
    batches = {}
    for name, loader in loaders.items():
        t0 = time.perf_counter()
        images, targets, target_lengths, texts = next(iter(loader))
        dt = time.perf_counter() - t0
        batches[name] = (images, texts)
        # codificarea și decodificarea trebuie să refacă exact textul
        decoded, start = [], 0
        for length in target_lengths.tolist():
            decoded.append(charset.decode(targets[start:start + length].tolist()))
            start += length
        assert decoded == texts, 'Codificarea etichetelor nu este reversibilă!'
        print(f'{name:5s}: imagini {tuple(images.shape)} {images.dtype}, valori în [{images.min():.2f}, {images.max():.2f}], '
              f'ținte {tuple(targets.shape)}, lungimi {tuple(target_lengths.shape)} | {len(loader)} batch-uri | '
              f'primul batch în {dt:.2f} s | exemple: {texts[:4]}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 6
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 2.6))
    show_batch(axes[0], *batches['train'], info['mean'], info['std'], 'train')
    show_batch(axes[1], *batches['val'], info['mean'], info['std'], 'val')
    fig.suptitle('Exemple după pre-procesare (rândul de sus: cu augmentări)')
    fig.tight_layout()
    path = out_dir / 'dataset_samples.png'
    fig.savefig(path, dpi=120)
    print(f'\nExemple salvate în {path}')


if __name__ == '__main__':  # necesar pe Windows când num_workers > 0
    main()
