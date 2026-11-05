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

## Structura (planificată)

```
.
├── data/              # setul de date IAM (NU este urmărit de git, vezi mai jos)
├── checkpoints/       # stările salvate ale antrenării (NU sunt urmărite de git)
├── src/
│   ├── dataset.py     # Dataset + DataLoader + transformări, împărțirea train/val/test
│   ├── check_dataset.py  # verificarea încărcării datelor (statistici, batch-uri, exemple)
│   ├── model.py       # arhitectura CRNN
│   ├── train.py       # antrenarea, salvarea și reluarea din checkpoint
│   ├── test.py        # evaluarea pe setul de test (CER, WER)
│   ├── plots.py       # graficele de antrenare / validare / testare
│   └── generative/    # modelul generativ pentru augmentare
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

Instrucțiunile de antrenare și testare vor fi adăugate odată cu componentele următoare (etapa S9).
