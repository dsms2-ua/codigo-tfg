import torch
import torch.nn as nn

from my_utils.preprocessing import IMG_HEIGHT, NUM_CHANNELS


# Copia del original
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        layers = [
            nn.Conv2d(NUM_CHANNELS, 8, (10, 2), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 8, (8, 5), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 1)),
        ]
        self.backbone = nn.Sequential(*layers)
        self.width_reduction = 2
        self.height_reduction = 2 ** 2
        self.out_channels = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class Encoder(nn.Module):
    ENC_DIM = 512  # 256 × 2 por ser bidireccional

    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = CNN()
        # Exponer atributos del CNN para que evaluate() siga funcionando igual
        self.width_reduction = self.cnn.width_reduction
        self.height_reduction = self.cnn.height_reduction

        input_size = self.cnn.out_channels * (IMG_HEIGHT // self.cnn.height_reduction)
        self.blstm = nn.LSTM(
            input_size,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.cnn(x)                                         # (B, 8, H//4, W//2)
        b = x.size(0)
        input_size = self.cnn.out_channels * (IMG_HEIGHT // self.cnn.height_reduction)
        x = x.permute(0, 3, 2, 1).contiguous()                 # (B, W//2, H//4, 8)
        x = x.reshape(b, -1, input_size)                       # (B, T, input_size)
        x, _ = self.blstm(x)                                   # (B, T, 512)
        x = self.dropout(x)
        return x                                                # (B, T, 512)


# Mecanismo de atención
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int = 512, dec_dim: int = 512, attn_dim: int = 256):
        super(BahdanauAttention, self).__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, enc_out: torch.Tensor, dec_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # enc_out:   (B, T, enc_dim)
        # dec_hidden:(B, dec_dim)
        energy = self.v(
            torch.tanh(
                self.W_enc(enc_out) + self.W_dec(dec_hidden).unsqueeze(1)
            )
        ).squeeze(-1)                                           # (B, T)
        weights = torch.softmax(energy, dim=-1)                # (B, T)
        context = (weights.unsqueeze(-1) * enc_out).sum(1)    # (B, enc_dim)
        return context, weights


# Decoder con atención
class Decoder(nn.Module):
    DEC_DIM = 512

    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super(Decoder, self).__init__()
        enc_dim = Encoder.ENC_DIM
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(enc_dim=enc_dim, dec_dim=self.DEC_DIM)
        self.rnn_cell = nn.LSTMCell(embed_dim + enc_dim, self.DEC_DIM)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(self.DEC_DIM + enc_dim, vocab_size)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.DEC_DIM, device=device)
        c = torch.zeros(batch_size, self.DEC_DIM, device=device)
        return h, c

    def forward_step(
        self,
        token: torch.Tensor,        # (B,)
        h: torch.Tensor,            # (B, DEC_DIM)
        c: torch.Tensor,            # (B, DEC_DIM)
        enc_out: torch.Tensor,      # (B, T, ENC_DIM)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embed(token)                                 # (B, embed_dim)
        context, attn_w = self.attention(enc_out, h)           # (B, enc_dim)
        h, c = self.rnn_cell(torch.cat([emb, context], dim=-1), (h, c))
        h = self.dropout(h)
        logit = self.fc_out(torch.cat([h, context], dim=-1))   # (B, vocab_size)
        return logit, h, c, attn_w


class Seq2SeqCRNN(nn.Module):
    def __init__(self, vocab_size: int):
        super(Seq2SeqCRNN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=vocab_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_step(
        self,
        token: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        enc_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder.forward_step(token, h, c, enc_out)

    def init_decoder_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.init_hidden(batch_size, device)