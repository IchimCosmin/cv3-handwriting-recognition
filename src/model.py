"""Arhitectura rețelei: CRNN (Convolutional Recurrent Neural Network) pentru recunoașterea cuvintelor scrise de mână.

Ideea: imaginea unui cuvânt este citită de la stânga la dreapta ca o secvență de coloane.

1. **CNN** (extractorul de trăsături): straturi convoluționale care transformă imaginea ``[B, 1, 64, 256]`` într-o hartă de
   trăsături ``[B, 512, 4, 64]``. Înălțimea scade de 16 ori (64 -> 4), lățimea doar de 4 ori (256 -> 64), astfel încât
   rămân 64 de coloane, fiecare descriind o fâșie verticală de 4 pixeli din imagine (cu un câmp receptiv mult mai larg).
2. **Map-to-sequence**: fiecare coloană (512 canale x 4 rânduri = 2.048 de valori) devine un vector al secvenței,
   proiectat liniar la 256 de dimensiuni. Rezultă o secvență de T = 64 de pași de timp.
3. **RNN**: un LSTM bidirecțional cu 2 straturi citește secvența în ambele sensuri, ca fiecare pas să "vadă" contextul
   literelor vecine (stânga și dreapta).
4. **Clasificator**: un strat liniar dă, pentru fiecare pas de timp, scorurile celor ``num_classes`` clase
   (caracterele alfabetului + simbolul *blank* al CTC, cu indexul 0), transformate în log-probabilități.

Ieșirea ``[T, B, num_classes]`` este exact formatul cerut de ``nn.CTCLoss``. Funcția de cost CTC (Connectionist Temporal
Classification) aliniază automat cei 64 de pași cu caracterele etichetei, fără a avea nevoie de poziția fiecărei litere.

Rulare pentru afișarea arhitecturii (forma tensorului după fiecare strat și numărul de parametri):
    python -m src.model
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.dataset import IMG_HEIGHT, IMG_WIDTH


class ConvBlock(nn.Sequential):
    """Convoluție 3 x 3 + Batch Normalization + ReLU.

    - ``Conv2d`` 3 x 3, padding 1: învață filtre locale (margini, curburi, bucle ale literelor); păstrează H și W.
      Nu are bias, pentru că BatchNorm care urmează are propriul termen de deplasare.
    - ``BatchNorm2d``: normalizează fiecare canal pe batch (medie 0, varianță 1), apoi aplică o scalare și o deplasare
      învățate; stabilizează și accelerează antrenarea.
    - ``ReLU``: neliniaritatea max(0, x); fără ea, straturile succesive s-ar reduce la o singură transformare liniară.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CRNN(nn.Module):
    """CNN + LSTM bidirecțional + clasificator pe caractere, antrenat cu CTC.

    Args:
        num_classes: numărul de clase la ieșire = numărul de caractere + 1 (blank), adică ``len(charset)``.
        img_height: înălțimea imaginilor de intrare (trebuie să fie multiplu de 16).
        rnn_hidden: dimensiunea stării ascunse a LSTM-ului (pe fiecare sens) și a proiecției de intrare.
        rnn_layers: numărul de straturi LSTM suprapuse.
        dropout: probabilitatea de dropout (după CNN, între straturile LSTM și înainte de clasificator).
    """

    def __init__(self, num_classes: int, img_height: int = IMG_HEIGHT, rnn_hidden: int = 256,
                 rnn_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        assert img_height % 16 == 0, 'Înălțimea imaginii trebuie să fie multiplu de 16 (4 straturi de pooling pe înălțime)'
        self.config = dict(num_classes=num_classes, img_height=img_height, rnn_hidden=rnn_hidden,
                           rnn_layers=rnn_layers, dropout=dropout)

        # 1. CNN: [B, 1, 64, 256] -> [B, 512, 4, 64]
        #    MaxPool 2 x 2 înjumătățește înălțimea și lățimea; MaxPool (2, 1) doar înălțimea, ca să păstrăm destui pași
        #    de timp pe orizontală (64 de pași pentru cuvinte de cel mult ~20 de caractere).
        self.cnn = nn.Sequential(
            ConvBlock(1, 64),                       # [B,  64, 64, 256]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,  64, 32, 128]
            ConvBlock(64, 128),                     # [B, 128, 32, 128]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, 16,  64]
            ConvBlock(128, 256),                    # [B, 256, 16,  64]
            ConvBlock(256, 256),                    # [B, 256, 16,  64]
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [B, 256, 8, 64]
            ConvBlock(256, 512),                    # [B, 512,  8,  64]
            ConvBlock(512, 512),                    # [B, 512,  8,  64]
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [B, 512, 4, 64]
            nn.Dropout2d(dropout),                  # elimină canale întregi aleator (regularizare)
        )
        feature_height = img_height // 16
        # 2. Map-to-sequence: fiecare coloană a hărții de trăsături (512 x 4 valori) -> vector de rnn_hidden dimensiuni
        self.proj = nn.Linear(512 * feature_height, rnn_hidden)
        # 3. LSTM bidirecțional: un LSTM citește secvența de la stânga la dreapta, altul de la dreapta la stânga;
        #    ieșirile lor sunt concatenate (2 * rnn_hidden). Porțile LSTM (input, forget, output) decid ce informație
        #    se păstrează în memoria celulei de la un pas la altul.
        self.rnn = nn.LSTM(rnn_hidden, rnn_hidden, num_layers=rnn_layers, bidirectional=True,
                           dropout=dropout if rnn_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        # 4. Clasificator: scorurile tuturor claselor (caractere + blank) pentru fiecare pas de timp
        self.classifier = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 1, H, W] -> log-probabilități [T, B, num_classes] (T = W / 4)."""
        features = self.cnn(images)                      # [B, C, H', T]
        b, c, h, t = features.shape
        seq = features.permute(3, 0, 1, 2).reshape(t, b, c * h)  # [T, B, C * H']: o coloană = un pas de timp
        seq = self.proj(seq)                             # [T, B, rnn_hidden]
        seq, _ = self.rnn(seq)                           # [T, B, 2 * rnn_hidden]
        logits = self.classifier(self.dropout(seq))      # [T, B, num_classes]
        return logits.log_softmax(dim=2)                 # log-probabilități, cerute de nn.CTCLoss

    @classmethod
    def from_config(cls, config: dict) -> 'CRNN':
        """Reconstruiește modelul din configurația salvată în checkpoint."""
        return cls(**config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: CRNN, height: int = IMG_HEIGHT, width: int = IMG_WIDTH) -> str:
    """Tabel cu forma tensorului după fiecare strat și numărul de parametri ai fiecărui strat."""
    lines = [f"{'Strat':<34}{'Formă ieșire':>24}{'Parametri':>12}"]
    x = torch.zeros(2, 1, height, width)
    model.eval()
    with torch.no_grad():
        for i, layer in enumerate(model.cnn):
            x = layer(x)
            name = f'cnn[{i}] ' + (layer[0].__class__.__name__ + f' {layer[0].in_channels}->{layer[0].out_channels} + BN + ReLU'
                                   if isinstance(layer, ConvBlock) else layer.__class__.__name__)
            lines.append(f'{name:<34}{str(list(x.shape)):>24}{count_parameters(layer):>12,}')
        b, c, h, t = x.shape
        x = x.permute(3, 0, 1, 2).reshape(t, b, c * h)
        lines.append(f"{'map-to-sequence (reshape)':<34}{str(list(x.shape)):>24}{0:>12,}")
        x = model.proj(x)
        lines.append(f"{'proj Linear':<34}{str(list(x.shape)):>24}{count_parameters(model.proj):>12,}")
        x, _ = model.rnn(x)
        lines.append(f"{'rnn BiLSTM x' + str(model.rnn.num_layers):<34}{str(list(x.shape)):>24}{count_parameters(model.rnn):>12,}")
        x = model.classifier(model.dropout(x))
        lines.append(f"{'dropout + classifier Linear':<34}{str(list(x.shape)):>24}{count_parameters(model.classifier):>12,}")
    lines.append(f"{'TOTAL':<34}{'':>24}{count_parameters(model):>12,}")
    return '\n'.join(lines)


if __name__ == '__main__':
    # Exemplu cu un alfabet de 79 de caractere + blank (numărul real rezultă din train, vezi check_dataset)
    crnn = CRNN(num_classes=80)
    print(crnn)
    print()
    print(describe(crnn))
    print('\nIntrare [B, 1, 64, 256] -> ieșire [T, B, clase]: T = 64 de pași de timp; fiecare pas acoperă 4 pixeli pe orizontală.')
