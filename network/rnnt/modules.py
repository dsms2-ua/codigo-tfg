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


# Equivalente al encoder de CTC
class TranscriptionEncoder(nn.Module):
    ENC_DIM = 512  # 256 × 2 por ser bidireccional

    def __init__(self):
        super(TranscriptionEncoder, self).__init__()
        self.cnn = CNN()
        self.width_reduction = self.cnn.width_reduction

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
        x = self.cnn(x)                                          # (B, 8, H//4, W//2)
        b = x.size(0)
        input_size = self.cnn.out_channels * (IMG_HEIGHT // self.cnn.height_reduction)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.reshape(b, -1, input_size)                        # (B, T, input_size)
        x, _ = self.blstm(x)                                    # (B, T, 512)
        x = self.dropout(x)
        return x                                                 # (B, T, ENC_DIM)


# No tiene atención, LSTM autoregresivo
class PredictionNetwork(nn.Module):
    PRED_DIM = 512

    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super(PredictionNetwork, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            self.PRED_DIM,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )
        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        tokens: torch.Tensor,          # (B, U)
        hidden: tuple = None,
    ) -> tuple[torch.Tensor, tuple]:
        emb = self.embed(tokens)        # (B, U, embed_dim)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        return out, hidden              # (B, U, PRED_DIM), hidden_state


class JointNetwork(nn.Module):
    JOINT_DIM = 512

    def __init__(self, vocab_size: int):
        super(JointNetwork, self).__init__()
        enc_dim = TranscriptionEncoder.ENC_DIM
        pred_dim = PredictionNetwork.PRED_DIM
        self.fc_enc = nn.Linear(enc_dim, self.JOINT_DIM)
        self.fc_pred = nn.Linear(pred_dim, self.JOINT_DIM)
        self.fc_out = nn.Linear(self.JOINT_DIM, vocab_size)

    def forward(
        self,
        enc_out: torch.Tensor,     # (B, T, ENC_DIM)
        pred_out: torch.Tensor,    # (B, U, PRED_DIM)
    ) -> torch.Tensor:
        enc_exp = enc_out.unsqueeze(2)              # (B, T, 1, ENC_DIM)
        pred_exp = pred_out.unsqueeze(1)            # (B, 1, U, PRED_DIM)
        h = torch.tanh(
            self.fc_enc(enc_exp) + self.fc_pred(pred_exp)
        )                                           # (B, T, U, JOINT_DIM)
        return self.fc_out(h)                       # (B, T, U, vocab_size)



class RNNTModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(RNNTModel, self).__init__()
        self.encoder = TranscriptionEncoder()
        self.prediction = PredictionNetwork(vocab_size=vocab_size)
        self.joint = JointNetwork(vocab_size=vocab_size)