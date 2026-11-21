"""Antrenarea modelului CRNN pe IAM, cu validare după fiecare epocă, grafice și checkpoint-uri.

Rulare (din folderul proiectului):
    python -m src.train --data_dir data --epochs 60
Reluarea unei antrenări întrerupte (aceiași hiperparametri, ordinea datelor și starea RNG sunt restaurate):
    python -m src.train --data_dir data --epochs 60 --resume auto
Continuarea unei antrenări terminate, cu mai multe epoci:
    python -m src.train --data_dir data --epochs 80 --resume checkpoints/crnn/last.pt

Fișierele rulării (în ``--run_dir``, implicit ``checkpoints/crnn``):
- ``last.pt``  - checkpoint-ul ultimei epoci terminate (pentru reluare);
- ``best.pt``  - checkpoint-ul cu cel mai mic CER pe validare (folosit la testare);
- ``history.json`` - metricile pe epoci; ``training_curves.png`` - graficele, actualizate după fiecare epocă.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from src.dataset import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import CRNN, count_parameters
from src.plots import plot_history
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description='Antrenarea CRNN + CTC pe IAM words')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--run_dir', default='checkpoints/crnn', help='folderul cu checkpoint-uri, istoric și grafice')
    p.add_argument('--epochs', type=int, default=60, help='numărul total de epoci (inclusiv cele deja făcute la reluare)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3, help='rata de învățare inițială (AdamW)')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=5.0, help='norma maximă a gradientului')
    p.add_argument('--lr_patience', type=int, default=3, help='epoci fără îmbunătățire a CER val până la înjumătățirea lr')
    p.add_argument('--early_stopping', type=int, default=12, help='epoci fără îmbunătățire până la oprire (0 = fără)')
    p.add_argument('--rnn_hidden', type=int, default=256)
    p.add_argument('--rnn_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.25)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--deterministic', action='store_true', help='algoritmi cuDNN determiniști (mai lent)')
    p.add_argument('--no_amp', action='store_true', help='dezactivează precizia mixtă (float16) pe GPU')
    p.add_argument('--resume', default=None, help="checkpoint de la care se reia antrenarea ('auto' = <run_dir>/last.pt)")
    p.add_argument('--max_batches', type=int, default=None, help='limitează batch-urile pe epocă (doar pentru teste rapide)')
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not args.no_amp

    resume_path = run_dir / 'last.pt' if args.resume == 'auto' else (Path(args.resume) if args.resume else None)
    checkpoint = load_checkpoint(resume_path) if resume_path else None
    if checkpoint is not None:
        # la reluare, seed-ul și arhitectura sunt cele din checkpoint (altfel împărțirea datelor și ponderile nu s-ar potrivi)
        args.seed = checkpoint['train_config']['seed']

    set_seed(args.seed, args.deterministic)
    loaders, charset, info = get_dataloaders(args.data_dir, batch_size=args.batch_size,
                                             num_workers=args.num_workers, seed=args.seed)
    train_generator = loaders['train'].generator

    model_config = checkpoint['model_config'] if checkpoint else dict(
        num_classes=len(charset), rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, dropout=args.dropout)
    model = CRNN.from_config(model_config).to(device)

    # CTC: indexul 0 = blank; zero_infinity ignoră exemplele imposibil de aliniat (etichetă mai lungă decât T)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Rata de învățare se înjumătățește când CER pe validare nu mai scade timp de lr_patience epoci
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_cer', 'val_wer', 'lr', 'seconds']}
    best_val_cer, best_epoch, start_epoch = float('inf'), 0, 1
    if checkpoint is not None:
        if checkpoint['charset'] != charset.chars:
            raise ValueError('Alfabetul din checkpoint diferă de cel al datelor curente (alt set de date sau alt seed?)')
        start_epoch = restore_training_state(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                                             scaler=scaler, train_generator=train_generator)
        history, best_val_cer, best_epoch = checkpoint['history'], checkpoint['best_val_cer'], checkpoint['best_epoch']
        print(f"Reluare din {resume_path}: epoca {checkpoint['epoch']} terminată, cel mai bun CER val "
              f"{100 * best_val_cer:.2f}% (epoca {best_epoch}), salvat la {checkpoint['saved_at']}")

    train_config = vars(args).copy()
    normalization = {'mean': info['mean'], 'std': info['std']}
    print(f'Dispozitiv: {device}' + (f' ({torch.cuda.get_device_name(0)}), AMP: {use_amp}' if device.type == 'cuda' else ''))
    print(f"Date: {info['n_words']} cuvinte; alfabet {len(charset.chars)} caractere + blank")
    print(f'Model CRNN: {count_parameters(model):,} parametri; epocile {start_epoch}-{args.epochs}')

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            # Ordinea batch-urilor și seed-urile augmentărilor din workeri depind doar de (seed, epocă): o antrenare
            # reluată din checkpoint vede exact aceleași date ca una neîntreruptă
            train_generator.manual_seed(args.seed + epoch)
            train_loss = train_one_epoch(model, loaders['train'], criterion, optimizer, scaler, device, use_amp=use_amp,
                                         grad_clip=args.grad_clip, max_batches=args.max_batches,
                                         desc=f'epoca {epoch} train')
            val = evaluate(model, loaders['val'], criterion, charset, device, use_amp=use_amp,
                           max_batches=args.max_batches, desc=f'epoca {epoch} val')
            scheduler.step(val['cer'])

            for key, value in [('epoch', epoch), ('train_loss', train_loss), ('val_loss', val['loss']),
                               ('val_cer', val['cer']), ('val_wer', val['wer']), ('lr', lr),
                               ('seconds', time.time() - start)]:
                history[key].append(value)
            improved = val['cer'] < best_val_cer
            if improved:
                best_val_cer, best_epoch = val['cer'], epoch

            save_checkpoint(run_dir / 'last.pt', model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, best_val_cer=best_val_cer, best_epoch=best_epoch, history=history,
                            model_config=model_config, train_config=train_config, charset=charset,
                            normalization=normalization, train_generator=train_generator)
            if improved:
                shutil.copyfile(run_dir / 'last.pt', run_dir / 'best.pt')
            (run_dir / 'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
            plot_history(history, run_dir / 'training_curves.png', best_epoch=best_epoch)

            examples = ', '.join(f'"{r}" -> "{p}"' for r, p in list(zip(val['references'], val['predictions']))[:3])
            print(f"epoca {epoch:3d} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
                  f"val CER {100 * val['cer']:6.2f}% | val WER {100 * val['wer']:6.2f}% | lr {lr:.1e} | "
                  f"{history['seconds'][-1]:.0f} s{' | *best*' if improved else ''} | {examples}")

            if args.early_stopping and epoch - best_epoch >= args.early_stopping:
                print(f'Oprire timpurie: CER pe validare nu s-a îmbunătățit în ultimele {args.early_stopping} epoci.')
                break
    except KeyboardInterrupt:
        print(f"\nAntrenare întreruptă. Ultima epocă terminată este salvată în {run_dir / 'last.pt'}; "
              f'reluați cu --resume auto.')
        return

    print(f'\nGata. Cel mai bun CER pe validare: {100 * best_val_cer:.2f}% (epoca {best_epoch}). '
          f"Testare: python -m src.test --checkpoint {run_dir / 'best.pt'} --data_dir {args.data_dir}")


if __name__ == '__main__':  # necesar pe Windows când num_workers > 0
    main()
