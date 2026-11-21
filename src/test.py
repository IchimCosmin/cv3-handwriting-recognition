"""Testarea modelului antrenat pe setul de test (autori nevăzuți la antrenare): CER, WER, analiza erorilor și grafice.

Rulare (din folderul proiectului):
    python -m src.test --checkpoint checkpoints/crnn/best.pt --data_dir data

Rezultate (în folderul checkpoint-ului sau în ``--out_dir``):
- ``test_metrics.json``     - loss, CER, WER, procentul de cuvinte recunoscute exact;
- ``test_predictions.csv``  - pentru fiecare cuvânt: id, autor, textul real, predicția, CER;
- ``training_curves.png``   - graficele antrenare + validare, cu rezultatele pe test adăugate;
- ``test_analysis.png``     - CER în funcție de lungimea cuvântului și distribuția erorilor;
- ``test_examples.png``     - exemple de predicții corecte și greșite.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from src.checkpoint import load_checkpoint
from src.dataset import get_dataloaders, load_image
from src.engine import evaluate
from src.model import CRNN
from src.plots import plot_examples, plot_history, plot_test_analysis
from src.utils import character_error_rate


def parse_args():
    p = argparse.ArgumentParser(description='Testarea CRNN pe IAM words')
    p.add_argument('--checkpoint', default='checkpoints/crnn/best.pt')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--split', default='test', choices=['test', 'val'], help='subsetul evaluat (implicit test)')
    p.add_argument('--out_dir', default=None, help='implicit: folderul checkpoint-ului')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--n_examples', type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_checkpoint(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aceeași împărțire, același alfabet și aceeași normalizare ca la antrenare (seed-ul este salvat în checkpoint)
    seed = checkpoint['train_config']['seed']
    loaders, charset, info = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
                                             seed=seed, augment=False)
    if charset.chars != checkpoint['charset']:
        raise ValueError('Alfabetul datelor diferă de cel din checkpoint (alt set de date sau alt seed?)')
    if abs(info['mean'] - checkpoint['normalization']['mean']) > 1e-6:
        raise ValueError('Normalizarea datelor diferă de cea de la antrenare')

    model = CRNN.from_config(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Model din {args.checkpoint}: epoca {checkpoint['epoch']}, CER val {100 * checkpoint['best_val_cer']:.2f}% "
          f"(cea mai bună epocă: {checkpoint['best_epoch']})")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    result = evaluate(model, loaders[args.split], criterion, charset, device, desc=args.split)

    samples = loaders[args.split].dataset.samples  # aceeași ordine ca predicțiile (shuffle=False)
    rows = [dict(word_id=s.word_id, writer=s.writer, text=ref, prediction=pred, cer=character_error_rate([ref], [pred]))
            for s, ref, pred in zip(samples, result['references'], result['predictions'])]
    exact = sum(r['text'] == r['prediction'] for r in rows) / len(rows)
    metrics = {'split': args.split, 'n_words': len(rows), 'n_writers': len({r['writer'] for r in rows}),
               'loss': result['loss'], 'cer': result['cer'], 'wer': result['wer'], 'exact_match': exact,
               'checkpoint': str(args.checkpoint), 'epoch': checkpoint['epoch']}

    print(f"\n=== Rezultate pe {args.split}: {metrics['n_words']} cuvinte, {metrics['n_writers']} autori ===")
    print(f"loss CTC {metrics['loss']:.4f} | CER {100 * metrics['cer']:.2f}% | WER {100 * metrics['wer']:.2f}% | "
          f'cuvinte recunoscute exact {100 * exact:.2f}%')

    prefix = '' if args.split == 'test' else 'val_'
    (out_dir / f'{prefix}test_metrics.json').write_text(json.dumps(metrics, indent=1), encoding='utf-8')
    with open(out_dir / f'{prefix}test_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.split == 'test':
        # best.pt conține istoricul doar până la epoca lui; history.json (scris după fiecare epocă) îl are complet
        history_file = Path(args.checkpoint).parent / 'history.json'
        history = json.loads(history_file.read_text(encoding='utf-8')) if history_file.exists() else checkpoint['history']
        plot_history(history, out_dir / 'training_curves.png', test_metrics=metrics, best_epoch=checkpoint['best_epoch'])
    plot_test_analysis(rows, out_dir / f'{prefix}test_analysis.png')

    # exemple: jumătate greșite, jumătate corecte, alese aleator (seed fix)
    rng = random.Random(seed)
    wrong = [i for i, r in enumerate(rows) if r['text'] != r['prediction']]
    right = [i for i, r in enumerate(rows) if r['text'] == r['prediction']]
    half = args.n_examples // 2
    chosen = rng.sample(wrong, min(half, len(wrong))) + rng.sample(right, min(args.n_examples - half, len(right)))
    plot_examples([load_image(samples[i].path) for i in chosen], [rows[i]['text'] for i in chosen],
                  [rows[i]['prediction'] for i in chosen], out_dir / f'{prefix}test_examples.png',
                  title=f'Predicții pe {args.split} (verde = corect, roșu = greșit)')
    print(f'Rezultatele și graficele sunt salvate în {out_dir}')


if __name__ == '__main__':  # necesar pe Windows când num_workers > 0
    main()
