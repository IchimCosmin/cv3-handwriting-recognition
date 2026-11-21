# Recunoașterea automată a cuvintelor scrise de mână

Proiect de semestru pentru laboratorul **CV 3**: un model de rețea neuronală care primește imaginea unui cuvânt scris de mână și returnează textul acestuia (*Handwritten Text Recognition*, HTR offline).

## Descrierea proiectului

| | |
| --- | --- |
| **Sarcina** | recunoașterea cuvintelor izolate scrise de mână (imagine → text, ieșire de lungime variabilă) |
| **Set de date** | [IAM Handwriting Database 3.0](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database): 115.320 de cuvinte etichetate, scrise de 657 de autori |
| **Model** | CRNN: extractor convoluțional (CNN) + LSTM bidirecțional + strat de clasificare pe caractere, antrenat cu funcția de cost CTC |
| **Metrica** | CER (*Character Error Rate*) - principală; WER (*Word Error Rate*) - secundară |
| **Augmentare generativă** | imagini sintetice de cuvinte generate de un model condiționat pe text (GAN sau VAE), adăugate doar în setul de antrenare |
| **Framework** | PyTorch |
| **Hardware** | NVIDIA GeForce RTX 5070 Ti (16 GB) |

Împărțirea în train / val / test se face **pe autori** (niciun autor nu apare în două subseturi), astfel încât modelul să fie evaluat pe scrisuri pe care nu le-a văzut la antrenare.

## Structura

```
.
├── data/              # setul de date IAM (NU este urmărit de git, vezi mai jos)
├── checkpoints/       # checkpoint-uri, istoricul și graficele antrenării (NU sunt urmărite de git)
├── src/
│   ├── dataset.py     # Dataset + DataLoader + transformări, împărțirea train/val/test
│   ├── check_dataset.py  # verificarea încărcării datelor (statistici, batch-uri, exemple)
│   ├── model.py       # arhitectura CRNN (CNN + BiLSTM + clasificator, cost CTC)
│   ├── engine.py      # o epocă de antrenare și evaluarea (loss, CER, WER)
│   ├── train.py       # scriptul de antrenare: validare, checkpoint-uri, reluare, grafice
│   ├── test.py        # testarea pe setul de test (CER, WER, analiza erorilor)
│   ├── plots.py       # graficele de antrenare / validare / testare
│   ├── checkpoint.py  # salvarea și încărcarea progresului (model, optimizator, scheduler, epocă, RNG)
│   ├── utils.py       # seed, starea RNG, decodarea CTC, metricile CER / WER
│   └── generative/    # (S12) modelul generativ pentru augmentare
├── requirements.txt
└── README.md
```

| Etapă | Conținut |
| --- | --- |
| S4 | crearea repository-ului |
| S6 | setul de date și DataLoader-ul funcționale (`src/dataset.py`) |
| S9 | rularea end-to-end a modelului (antrenare, validare, testare, grafice, checkpoint) |
| S12 | augmentarea cu metoda generativă |
| S14 | prezentarea proiectului |

## Instalare

Proiectul folosește **Python 3.12**.

1. Creați și activați un mediu virtual:

   ```bash
   python3.12 -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. Instalați **PyTorch cu suport CUDA**. Plăcile din seria RTX 50 (arhitectura Blackwell) au nevoie de PyTorch 2.7 sau mai nou, compilat pentru CUDA 12.8 sau mai nou. Pe Windows, pachetul `torch` de pe PyPI este doar pentru CPU, de aceea comanda de instalare se ia de pe [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (alegeți sistemul de operare, `Pip` și varianta CUDA), pentru versiunile `torch==2.14.1` și `torchvision==0.29.1`.

   Verificare:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

3. Instalați restul dependențelor:

   ```bash
   pip install -r requirements.txt
   ```

## Descărcarea setului de date

Setul de date **nu este inclus** în repository.

1. Înregistrați-vă (gratuit) pe [pagina IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).
2. De pe [pagina de descărcare](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database), descărcați:
   - `words.tgz` - imaginile cuvintelor;
   - `ascii.tgz` - adnotările: `words.txt` (textul fiecărui cuvânt) și `forms.txt` (autorul fiecărei pagini, folosit la împărțirea train/val/test).
3. Dezarhivați-le în folderul `data/` (comanda `tar` există și pe Windows 10/11):

   ```bash
   python -c "import os; [os.makedirs(d, exist_ok=True) for d in ('data/words', 'data/ascii')]"
   tar -xzf words.tgz -C data/words
   tar -xzf ascii.tgz -C data/ascii
   ```

   Structura rezultată:

   ```
   data/
   ├── words/      # a01/a01-000u/a01-000u-00-00.png, ... (din words.tgz)
   └── ascii/      # words.txt, forms.txt, ... (din ascii.tgz)
   ```

   Codul găsește fișierele și dacă arhivele au fost dezarhivate într-un subfolder suplimentar (de exemplu `data/ascii/ascii/words.txt`).

Setul de date poate fi folosit doar în scopuri de cercetare non-comerciale. Referință: U. Marti, H. Bunke, *The IAM-database: An English Sentence Database for Off-line Handwriting Recognition*, International Journal on Document Analysis and Recognition, vol. 5, pp. 39-46, 2002.

## Încărcarea și pre-procesarea datelor (`src/dataset.py`)

| Pas | Ce face |
| --- | --- |
| Adnotări | citește `words.txt`; elimină cuvintele cu segmentare marcată `err` și cele 2 imagini corupte cunoscute (`a01-117-05-02`, `r06-022-03-05`) |
| Împărțire | **pe autori**, 80% / 10% / 10% (train / val / test), cu seed fix (42): niciun autor nu apare în două subseturi |
| Alfabet | construit doar din train; indexul 0 este simbolul *blank* pentru CTC; cuvintele din val/test cu caractere necunoscute sunt eliminate (numărul lor este afișat) |
| Imagini | grayscale, redimensionate fără deformare astfel încât să încapă în 64 × 256 px, completate cu alb (padding) la dreapta și jos |
| Normalizare | media și deviația standard calculate pe 2.000 de imagini din train (nu din val/test) |
| Augmentări (doar train) | rotație ±2°, înclinare (shear) ±10°, translație și scalare ușoare, variații de luminozitate și contrast |
| DataLoader | batch-uri cu imagini `[B, 1, 64, 256]`; etichetele sunt concatenate, cu lungimile lor separat (formatul cerut de `nn.CTCLoss`) |

Utilizare:

```python
from src.dataset import get_dataloaders

loaders, charset, info = get_dataloaders('data', batch_size=64, num_workers=4)
for images, targets, target_lengths, texts in loaders['train']:
    ...
```

## Rulare

Verificarea încărcării datelor (statistici pentru fiecare subset, verificarea autorilor disjuncți, forma batch-urilor, codificarea etichetelor; salvează exemple în `outputs/dataset_samples.png`):

```bash
python -m src.check_dataset --data_dir data
```

Afișarea arhitecturii modelului (forma tensorului după fiecare strat și numărul de parametri):

```bash
python -m src.model
```

Antrenarea, testarea și reluarea sunt descrise mai jos.

## Modelul (`src/model.py`)

**CRNN** (*Convolutional Recurrent Neural Network*, Shi et al., 2016): imaginea cuvântului este citită de la stânga la dreapta ca o secvență de coloane. Rețeaua are **~7,7 milioane de parametri** (numărul exact depinde de mărimea alfabetului; vezi `python -m src.model`).

| # | Strat | Ieșire (pentru o imagine) | Rol |
| --- | --- | --- | --- |
| 0 | intrare | 1 × 64 × 256 | imaginea grayscale normalizată |
| 1 | Conv 3×3 (64) + BatchNorm + ReLU | 64 × 64 × 256 | filtre locale: margini, trăsături de bază ale scrisului |
| 2 | MaxPool 2×2 | 64 × 32 × 128 | reduce rezoluția, păstrează răspunsurile cele mai puternice |
| 3 | Conv 3×3 (128) + BN + ReLU | 128 × 32 × 128 | combinații de margini: curburi, bucle |
| 4 | MaxPool 2×2 | 128 × 16 × 64 | |
| 5-6 | 2 × Conv 3×3 (256) + BN + ReLU | 256 × 16 × 64 | părți de litere |
| 7 | MaxPool (2, 1) | 256 × 8 × 64 | reduce doar înălțimea: păstrăm 64 de poziții pe orizontală |
| 8-9 | 2 × Conv 3×3 (512) + BN + ReLU | 512 × 8 × 64 | trăsături de nivel înalt (litere, grupuri de litere) |
| 10 | MaxPool (2, 1) + Dropout2d | 512 × 4 × 64 | regularizare: elimină canale întregi aleator |
| 11 | map-to-sequence | 64 × 2048 | fiecare coloană (512 canale × 4 rânduri) devine un pas de timp |
| 12 | Linear 2048 → 256 | 64 × 256 | proiecția trăsăturilor coloanei |
| 13 | BiLSTM, 2 straturi, 256 pe sens | 64 × 512 | context: fiecare pas vede literele din stânga și din dreapta |
| 14 | Dropout + Linear 512 → clase | 64 × (caractere + 1) | scorul fiecărui caracter (și al simbolului *blank*) la fiecare pas |
| 15 | log-softmax | 64 × (caractere + 1) | log-probabilități, cerute de CTC |

- **Conv + BatchNorm + ReLU**: convoluția învață filtre locale; BatchNorm normalizează fiecare canal pe batch (antrenare mai stabilă și mai rapidă); ReLU introduce neliniaritatea.
- **MaxPool**: păstrează maximul din fiecare fereastră; ultimele două reduc doar înălțimea, pentru a rămâne T = 64 de pași de timp (fiecare acoperă 4 pixeli pe orizontală), suficienți pentru cuvinte de până la ~20 de caractere.
- **LSTM bidirecțional**: două LSTM-uri citesc secvența în sensuri opuse; porțile LSTM (input, forget, output) controlează ce informație se păstrează de la un pas la altul.
- **CTC** (*Connectionist Temporal Classification*): funcția de cost însumează probabilitățile tuturor alinierilor posibile dintre cei 64 de pași și caracterele etichetei, deci nu este nevoie de poziția fiecărei litere în imagine. La predicție, decodarea *greedy* alege clasa cea mai probabilă la fiecare pas, unește caracterele repetate consecutiv și elimină *blank*-urile (`-hh-e-ll-ll-oo` → `hello`).

## Antrenarea (`src/train.py`, `src/engine.py`)

| | |
| --- | --- |
| Funcția de cost | `nn.CTCLoss` (blank = 0, `zero_infinity=True`), calculată în float32 |
| Optimizator | AdamW, lr = 1e-3, weight decay = 1e-4 |
| Rata de învățare | `ReduceLROnPlateau`: înjumătățită dacă CER pe validare nu scade 3 epoci la rând |
| Stabilitate | tăierea normei gradientului la 5 (LSTM); precizie mixtă (float16) pe GPU |
| Validare | după fiecare epocă: loss, CER și WER pe setul de validare |
| Modelul păstrat | cel cu cel mai mic CER pe validare (`best.pt`) |
| Oprire timpurie | după 12 epoci fără îmbunătățirea CER pe validare |
| Reproductibilitate | seed fix (42); ordinea datelor și augmentările fiecărei epoci depind doar de (seed, epocă) |

```bash
python -m src.train --data_dir data --epochs 60
```

Opțiuni utile: `--batch_size`, `--lr`, `--num_workers` (pe Windows, 2-4), `--run_dir` (implicit `checkpoints/crnn`), `--no_amp`, `--deterministic`, `--max_batches 20` (test rapid al întregului lanț).

După fiecare epocă, în `checkpoints/crnn/` sunt actualizate: `last.pt`, `best.pt`, `history.json` și `training_curves.png`.

## Checkpoint și reluarea antrenării (`src/checkpoint.py`)

| Cheie | Conținut |
| --- | --- |
| `epoch` | ultima epocă terminată (antrenarea se reia de la următoarea) |
| `model_state` | ponderile rețelei |
| `optimizer_state` | starea AdamW (momentele de ordinul 1 și 2, pasul curent) |
| `scheduler_state` | starea `ReduceLROnPlateau` (cea mai bună valoare, epocile fără îmbunătățire, lr curent) |
| `scaler_state` | starea `GradScaler` (precizia mixtă) |
| `rng_state` | starea generatoarelor aleatoare: Python, NumPy, PyTorch CPU și GPU, generatorul DataLoader-ului |
| `best_val_cer`, `best_epoch`, `history` | cel mai bun CER pe validare și istoricul metricilor |
| `model_config`, `train_config` | hiperparametrii modelului și ai antrenării (inclusiv seed-ul) |
| `charset`, `normalization` | alfabetul și media / deviația standard, necesare la testare |

Salvarea se face într-un fișier temporar, apoi acesta este redenumit, deci o întrerupere în timpul salvării nu strică checkpoint-ul anterior. Reluarea unei antrenări întrerupte (de exemplu cu Ctrl+C):

```bash
python -m src.train --data_dir data --epochs 60 --resume auto
```

`--resume auto` folosește `<run_dir>/last.pt`; se poate da și o cale explicită, iar `--epochs` poate fi mărit pentru a continua o antrenare terminată. Reluarea este exactă: o antrenare oprită după epoca 2 și reluată până la epoca 3 dă aceleași valori ale funcției de cost ca o antrenare neîntreruptă de 3 epoci (verificat cu `--deterministic`).

## Testarea (`src/test.py`)

```bash
python -m src.test --checkpoint checkpoints/crnn/best.pt --data_dir data
```

Modelul este evaluat pe setul de test (autori nevăzuți la antrenare), cu aceeași împărțire, același alfabet și aceeași normalizare ca la antrenare (luate din checkpoint). Rezultate, în folderul checkpoint-ului:

| Fișier | Conținut |
| --- | --- |
| `test_metrics.json` | loss CTC, CER, WER și procentul de cuvinte recunoscute exact |
| `test_predictions.csv` | pentru fiecare cuvânt: id, autor, textul real, predicția, CER |
| `training_curves.png` | loss train/validare, CER și WER pe validare și rata de învățare pe epoci, cu rezultatele pe test (linii punctate) |
| `test_analysis.png` | CER în funcție de lungimea cuvântului și distribuția CER pe cuvânt |
| `test_examples.png` | exemple de predicții corecte și greșite |

## Graficele (`src/plots.py`)

- `plot_history`: evoluția funcției de cost pe train și validare, CER / WER pe validare (cu epoca celui mai bun model marcată) și rata de învățare; actualizat după fiecare epocă și, la testare, completat cu rezultatele pe test;
- `plot_test_analysis`: erorile pe test în funcție de lungimea cuvântului;
- `plot_examples`: imagini de test cu eticheta reală și predicția.
