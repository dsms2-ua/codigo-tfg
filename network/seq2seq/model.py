import time
import random
import gc
from typing import List, Dict, Tuple, Generator

import torch
import pandas as pd
import torch.nn as nn
from torchinfo import summary

from network.seq2seq.modules import Seq2SeqCRNN
from my_utils.preprocessing import preprocess_audio, preprocess_label, IMG_HEIGHT, NUM_CHANNELS
from my_utils.metrics import compute_metrics
from my_utils.encoding_convertions import krnConverter


class Seq2SeqTrainedCRNN:
    """
    Wrapper de entrenamiento para el modelo Seq2Seq con atención de Bahdanau.

    Diferencias respecto a CTCTrainedCRNN:
    - Pérdida: CrossEntropyLoss en lugar de CTCLoss.
    - Entrenamiento: bucle autoregresivo con teacher forcing.
    - Vocabulario: requiere tokens <SOS> y <EOS> además de <PAD>.
    - Inferencia: decodificación greedy token a token hasta <EOS>.
    """

    def __init__(
        self,
        dictionaries: Tuple[Dict[str, int], Dict[int, str]],
        encoding: str,
        device: torch.device,
    ):
        self.w2i, self.i2w = dictionaries
        self.pad_index = self.w2i["<PAD>"]
        self.sos_index = self.w2i["<SOS>"]  # nuevo token requerido
        self.eos_index = self.w2i["<EOS>"]  # nuevo token requerido
        self.device = device
        self.krnParser = krnConverter(encoding=encoding)

        self.model = Seq2SeqCRNN(vocab_size=len(self.w2i))
        self.model.to(self.device)
        self.compile()
        self.summary()

    def summary(self):
        # Solo resumimos el encoder; el decoder se muestra aparte
        summary(self.model.encoder, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def compile(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # Ignoramos <PAD> en la pérdida (posiciones de relleno del target)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def on_train_begin(self, patience: int):
        self.logs = {"loss": [], "val_ser": [], "val_mv2h": [], "val_recon_ser": []}
        self.best_val_ser = float("inf")
        self.best_epoch = 0
        self.patience = patience

    def train_step(
        self,
        data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        x, xl, y, yl = data
        # y: (B, max_len) con <SOS> al inicio y <EOS> al final, relleno con <PAD>
        self.optimizer.zero_grad()

        enc_out = self.model.encode(x)          # (B, T, enc_dim)
        B = x.size(0)
        max_len = y.size(1)

        h, c = self.model.init_decoder_hidden(B, self.device)
        token = y[:, 0]                         # primer token: <SOS>
        loss = 0

        for t in range(1, max_len):
            logit, h, c, _ = self.model.decode_step(token, h, c, enc_out)
            # logit: (B, vocab_size)  |  y[:, t]: índice objetivo
            loss += self.criterion(logit, y[:, t])
            # Teacher forcing: con probabilidad p usamos la etiqueta real,
            # si no, el token predicho por el modelo
            use_teacher = random.random() < teacher_forcing_ratio
            token = y[:, t] if use_teacher else logit.argmax(dim=-1)

        loss = loss / (max_len - 1)
        loss.backward()
        self.optimizer.step()
        return loss

    def fit(
        self,
        train_data_generator: Generator,
        epochs: int,
        steps_per_epoch: int,
        val_data: Tuple[List[str], List[str]],
        test_data: Tuple[List[str], List[str]],
        patience: int,
        weights_path: str,
        logs_path: str,
    ):
        XFVal, YFVal = val_data
        XFTest, YFTest = test_data

        self.on_train_begin(patience=patience)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", end="", flush=True)
            start = time.time()

            self.model.train()
            for _ in range(steps_per_epoch):
                data = next(train_data_generator)
                loss = self.train_step(data)
            loss = loss.detach().cpu().item()
            self.logs["loss"].append(loss)
            print(f" - loss: {loss:.4f}", end="", flush=True)

            self.model.eval()
            metrics = self.evaluate(
                XFiles=XFVal,
                YFiles=YFVal,
                krnParser=self.krnParser,
                aux_name=str(weights_path).split("/")[-2],
                print_metrics=False,
            )
            for k, v in metrics.items():
                self.logs[f"val_{k}"].append(v)
                print(f" - val_{k}: {v:.2f}", end="", flush=True)

            print(f" - {round(time.time() - start)}s")

            if metrics["ser"] < self.best_val_ser:
                print(f"SER improved from {self.best_val_ser:.2f} to {metrics['ser']:.2f}")
                print(f"Saving weights to {weights_path}")
                self.best_val_ser = metrics["ser"]
                self.best_epoch = epoch
                self.patience = patience
                self.save(path=weights_path)
            else:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Stopped by early stopping on epoch: {epoch + 1}")
                    break

            del loss
            gc.collect()
            torch.cuda.empty_cache()

        self.load(path=weights_path)
        self.model.eval()
        print(f"Evaluating best model (epoch {self.best_epoch}) on test data")
        metrics = self.evaluate(
            XFiles=XFTest,
            YFiles=YFTest,
            krnParser=self.krnParser,
            aux_name=str(weights_path).split("/")[-2],
            print_random_samples=True,
        )
        for k, v in metrics.items():
            self.logs[f"test_{k}"] = v

        print(f"Saving logs to {logs_path}")
        self.save_logs(logs_path)

    def evaluate(
        self,
        XFiles: List[str],
        YFiles: List[str],
        krnParser: krnConverter,
        aux_name: str,
        print_metrics: bool = True,
        print_random_samples: bool = False,
        max_decode_length: int = 512,
    ) -> Dict[str, float]:
        Y = []
        YPRED = []

        with torch.no_grad():
            for xf, yf in zip(XFiles, YFiles):
                # Preprocesar audio (igual que CTC)
                x = preprocess_audio(
                    xf,
                    training=False,
                    width_reduction=self.model.encoder.width_reduction,
                )
                x = torch.from_numpy(x.copy()).float().unsqueeze(0).to(self.device)

                # Codificar
                enc_out = self.model.encode(x)          # (1, T, enc_dim)
                h, c = self.model.init_decoder_hidden(1, self.device)
                token = torch.tensor([self.sos_index], device=self.device)

                # Decodificación greedy token a token
                pred_tokens = []
                for _ in range(max_decode_length):
                    logit, h, c, _ = self.model.decode_step(token, h, c, enc_out)
                    token = logit.argmax(dim=-1)         # (1,)
                    idx = token.item()
                    if idx == self.eos_index:
                        break
                    if idx != self.pad_index:
                        pred_tokens.append(self.i2w[idx])

                YPRED.append(pred_tokens)

                # Preprocesar etiqueta (igual que CTC)
                y = preprocess_label(yf, training=False, w2i=self.w2i, krnParser=krnParser)
                Y.append(y)

        metrics = compute_metrics(Y, YPRED, encoding=krnParser.encoding, aux_name=aux_name)

        if print_metrics:
            for k, v in metrics.items():
                print(f"{k.upper()} (%): {v:.2f} - ", end="", flush=True)
            print(f"From {len(Y)} samples")

        if print_random_samples:
            index = random.randint(0, len(Y) - 1)
            print(f"Prediction - {YPRED[index]}")
            print(f"Ground truth - {Y[index]}")

        return metrics

    def save_logs(self, path: str):
        for k in self.logs.keys():
            if "test" not in k:
                self.logs[k].extend(["-", self.logs[k][self.best_epoch]])
            else:
                self.logs[k] = ["-"] * len(self.logs["loss"][:-1]) + [self.logs[k]]
        pd.DataFrame.from_dict(self.logs).to_csv(path, index=False)