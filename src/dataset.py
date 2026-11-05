"""Încărcarea și pre-procesarea setului de date IAM (cuvinte scrise de mână).

Componentele acestui modul:
- citirea adnotărilor (``ascii/words.txt``) și a autorilor fiecărei pagini (``ascii/forms.txt``);
- împărțirea pe autori în train / val / test (niciun autor nu apare în două subseturi);
- alfabetul și codificarea textului în indecși pentru funcția de cost CTC (indexul 0 = simbolul blank);
- transformările imaginilor: redimensionare cu păstrarea proporțiilor, completare (padding) la o dimensiune fixă,
  normalizare și, doar pentru train, augmentări geometrice și fotometrice ușoare;
- clasa ``IAMWords`` (``torch.utils.data.Dataset``) și funcția ``get_dataloaders`` care creează cele trei DataLoader-e.

Structura așteptată a folderului de date (vezi README):

    data/
    ├── words/   a01/a01-000u/a01-000u-00-00.png, ...   (din words.tgz)
    └── ascii/   words.txt, forms.txt                    (din ascii.tgz)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Imagini cunoscute ca fiind corupte în IAM (fișiere goale sau care nu pot fi citite)
BAD_SAMPLES = {'a01-117-05-02', 'r06-022-03-05'}

# Dimensiunea fixă a imaginilor de intrare: toate cuvintele sunt aduse la înălțimea de 64 px și completate la 256 px lățime
IMG_HEIGHT = 64
IMG_WIDTH = 256

# Proporțiile împărțirii pe autori: 80% train, 10% val, 10% test
SPLIT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class Sample:
    """O intrare din setul de date: calea imaginii, textul (eticheta) și autorul."""
    word_id: str
    path: Path
    text: str
    writer: str


# ---------------------------------------------------------------------------------------------------------------------
# Citirea adnotărilor
# ---------------------------------------------------------------------------------------------------------------------

def find_file(data_dir: Path, name: str) -> Path:
    """Caută un fișier de adnotări (words.txt / forms.txt) în ``data/ascii`` sau oriunde sub ``data/``
    (în funcție de cum a fost dezarhivat ascii.tgz)."""
    data_dir = Path(data_dir)
    direct = data_dir / 'ascii' / name
    if direct.exists():
        return direct
    found = sorted(data_dir.rglob(name))
    if not found:
        raise FileNotFoundError(f'Nu am găsit {name} în {data_dir}. Vezi instrucțiunile de descărcare din README.')
    return found[0]


def find_images_root(data_dir: Path) -> Path:
    """Folderul care conține subfolderele a01, a02, ... din words.tgz (``data/words`` sau un subfolder al lui)."""
    data_dir = Path(data_dir)
    for candidate in [data_dir / 'words', data_dir / 'words' / 'words', data_dir]:
        if (candidate / 'a01').is_dir():
            return candidate
    raise FileNotFoundError(f'Nu am găsit imaginile cuvintelor (folderul a01) în {data_dir}/words. Vezi README.')


def read_writers(forms_file: Path) -> dict[str, str]:
    """Citește ``forms.txt`` și întoarce dicționarul {id pagină: id autor}.

    Format linie: ``a01-000u 000 2 prt 7 5 52 36`` (primul câmp = pagina, al doilea = autorul).
    """
    writers = {}
    with open(forms_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            writers[parts[0]] = parts[1]
    return writers


def read_words(data_dir: Path, keep_bad_segmentation: bool = False) -> list[Sample]:
    """Citește ``words.txt`` și întoarce lista de cuvinte, cu autorul fiecăruia.

    Format linie: ``a01-000u-00-00 ok 154 408 768 27 51 AT A``
    (id cuvânt, rezultatul segmentării ok/err, prag de binarizare, x y w h, etichetă gramaticală, text).
    Cuvintele marcate ``err`` (segmentare posibil greșită) sunt eliminate, cu excepția cazului în care
    ``keep_bad_segmentation=True``. Sunt eliminate și cele 2 imagini corupte cunoscute.
    """
    data_dir = Path(data_dir)
    writers = read_writers(find_file(data_dir, 'forms.txt'))
    images_root = find_images_root(data_dir)
    samples = []
    with open(find_file(data_dir, 'words.txt'), encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split(' ')
            if len(parts) < 9:
                continue
            word_id, segmentation = parts[0], parts[1]
            if word_id in BAD_SAMPLES:
                continue
            if segmentation != 'ok' and not keep_bad_segmentation:
                continue
            text = ' '.join(parts[8:])  # textul este ultimul câmp (în cazuri rare poate conține spații)
            if not text:
                continue
            # a01-000u-00-00 -> words/a01/a01-000u/a01-000u-00-00.png
            p1, p2 = word_id.split('-')[:2]
            form_id = f'{p1}-{p2}'
            path = images_root / p1 / form_id / f'{word_id}.png'
            samples.append(Sample(word_id, path, text, writers[form_id]))
    return samples


# ---------------------------------------------------------------------------------------------------------------------
# Împărțirea în train / val / test
# ---------------------------------------------------------------------------------------------------------------------

def split_by_writer(samples: list[Sample], ratios=SPLIT_RATIOS, seed: int = 42) -> dict[str, list[Sample]]:
    """Împarte cuvintele în train / val / test după autor, cu seed fix (rezultat reproductibil).

    Autorii sunt amestecați aleator și împărțiți în proporțiile date; toate cuvintele unui autor ajung în același subset.
    Astfel, val și test măsoară performanța pe scrisuri pe care modelul nu le-a văzut la antrenare.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, 'Proporțiile trebuie să aibă suma 1'
    writers = sorted({s.writer for s in samples})
    random.Random(seed).shuffle(writers)
    n_train = int(round(ratios[0] * len(writers)))
    n_val = int(round(ratios[1] * len(writers)))
    groups = {
        'train': set(writers[:n_train]),
        'val': set(writers[n_train:n_train + n_val]),
        'test': set(writers[n_train + n_val:]),
    }
    return {name: [s for s in samples if s.writer in group] for name, group in groups.items()}


# ---------------------------------------------------------------------------------------------------------------------
# Alfabetul și codificarea textului pentru CTC
# ---------------------------------------------------------------------------------------------------------------------

class Charset:
    """Alfabetul modelului. Indexul 0 este rezervat simbolului *blank* al funcției de cost CTC."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}

    @classmethod
    def from_samples(cls, samples: list[Sample]) -> 'Charset':
        return cls(''.join(s.text for s in samples))

    def __len__(self):
        # numărul de clase la ieșirea rețelei = caracterele + blank
        return len(self.chars) + 1

    def covers(self, text: str) -> bool:
        return all(c in self.char_to_idx for c in text)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices) -> str:
        """Transformă indecșii înapoi în text (fără regulile CTC: nu elimină dublurile și blank-urile)."""
        return ''.join(self.chars[i - 1] for i in indices if i > 0)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.chars, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'Charset':
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))


# ---------------------------------------------------------------------------------------------------------------------
# Transformările imaginilor
# ---------------------------------------------------------------------------------------------------------------------

class ResizePad:
    """Redimensionează imaginea (grayscale) astfel încât să încapă în height x width fără a-i deforma proporțiile,
    apoi o completează cu alb (fundalul paginii) la dreapta și jos, până la dimensiunea fixă height x width."""

    def __init__(self, height: int = IMG_HEIGHT, width: int = IMG_WIDTH, fill: int = 255):
        self.height, self.width, self.fill = height, width, fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.height / h, self.width / w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new('L', (self.width, self.height), self.fill)
        canvas.paste(img, (0, 0))
        return canvas


def build_transforms(train: bool, mean: float, std: float, height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
    """Lanțul de transformări. Augmentările (doar pentru train) simulează variații ale scrisului:
    înclinare (shear), rotație și scalare ușoare, plus variații de contrast și luminozitate."""
    steps = [ResizePad(height, width)]
    if train:
        steps += [
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.05), scale=(0.9, 1.05), shear=(-10, 10), fill=255),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    steps += [
        transforms.ToTensor(),                         # [1, H, W], valori în [0, 1]
        transforms.Normalize(mean=[mean], std=[std]),  # statistici calculate doar pe train
    ]
    return transforms.Compose(steps)


def load_image(path: Path) -> Image.Image:
    """Citește imaginea ca grayscale (mod 'L')."""
    with Image.open(path) as img:
        return ImageOps.grayscale(img).copy()


def compute_mean_std(samples: list[Sample], n: int = 2000, seed: int = 42,
                     height: int = IMG_HEIGHT, width: int = IMG_WIDTH) -> tuple[float, float]:
    """Media și deviația standard a pixelilor, calculate pe un eșantion aleator de imagini din TRAIN
    (după redimensionare și padding, exact ca la antrenare). Val și test nu sunt folosite (fără data leakage)."""
    rng = random.Random(seed)
    subset = rng.sample(samples, min(n, len(samples)))
    resize = ResizePad(height, width)
    total, total_sq, count = 0.0, 0.0, 0
    for s in subset:
        x = np.asarray(resize(load_image(s.path)), dtype=np.float64) / 255.0
        total += x.sum()
        total_sq += (x ** 2).sum()
        count += x.size
    mean = total / count
    std = float(np.sqrt(total_sq / count - mean ** 2))
    return float(mean), std


# ---------------------------------------------------------------------------------------------------------------------
# Dataset și DataLoader
# ---------------------------------------------------------------------------------------------------------------------

class IAMWords(Dataset):
    """Dataset de cuvinte IAM. ``__getitem__`` întoarce (imagine [1, H, W], indecșii caracterelor, textul)."""

    def __init__(self, samples: list[Sample], charset: Charset, transform):
        # Cuvintele cu caractere care nu există în alfabet (construit doar din train) nu pot fi învățate/evaluate cu CTC
        self.samples = [s for s in samples if charset.covers(s.text)]
        self.n_dropped = len(samples) - len(self.samples)
        self.charset = charset
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = self.transform(load_image(s.path))
        target = torch.tensor(self.charset.encode(s.text), dtype=torch.long)
        return image, target, s.text


def collate_ctc(batch):
    """Grupează un batch pentru CTC: imaginile au aceeași dimensiune și se pot stivui; etichetele au lungimi diferite,
    deci sunt concatenate într-un singur vector, împreună cu lungimea fiecăreia (formatul cerut de nn.CTCLoss)."""
    images, targets, texts = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return images, targets, target_lengths, list(texts)


def get_dataloaders(data_dir='data', batch_size: int = 64, num_workers: int = 4, seed: int = 42,
                    augment: bool = True, height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
    """Creează DataLoader-ele de train / val / test și întoarce și alfabetul + informațiile despre împărțire.

    - împărțire pe autori 80/10/10, cu seed fix;
    - alfabetul și statisticile de normalizare se calculează doar pe train;
    - augmentările se aplică doar pe train; val și test folosesc doar redimensionare, padding și normalizare.
    """
    samples = read_words(data_dir)
    splits = split_by_writer(samples, seed=seed)
    charset = Charset.from_samples(splits['train'])
    mean, std = compute_mean_std(splits['train'], seed=seed, height=height, width=width)

    datasets = {
        name: IAMWords(split, charset, build_transforms(train=(name == 'train' and augment), mean=mean, std=std,
                                                         height=height, width=width))
        for name, split in splits.items()
    }

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_ctc,
                       pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    generator = torch.Generator().manual_seed(seed)  # ordinea batch-urilor de train este reproductibilă
    loaders = {
        'train': DataLoader(datasets['train'], shuffle=True, drop_last=True, generator=generator, **loader_args),
        'val': DataLoader(datasets['val'], shuffle=False, **loader_args),
        'test': DataLoader(datasets['test'], shuffle=False, **loader_args),
    }
    info = {
        'mean': mean, 'std': std,
        'n_words': {k: len(v) for k, v in datasets.items()},
        'n_writers': {k: len({s.writer for s in v}) for k, v in splits.items()},
        'n_dropped_unknown_chars': {k: v.n_dropped for k, v in datasets.items()},
    }
    return loaders, charset, info


def label_stats(samples: list[Sample]) -> dict:
    """Statistici despre etichete, utile pentru verificare (lungimi, caractere frecvente)."""
    lengths = [len(s.text) for s in samples]
    return {
        'max_len': max(lengths), 'mean_len': float(np.mean(lengths)),
        'most_common_chars': Counter(''.join(s.text for s in samples)).most_common(10),
    }
