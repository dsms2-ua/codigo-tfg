import time
import random
import gc
from typing import List, Dict, Tuple, Generator

import torch
import pandas as pd
import torch.nn as nn
import torchaudio
from torchinfo import summary

from network.rnnt.modules import RNNTModel
from my_utils.preprocessing import preprocess_audio, preprocess_label, IMG_HEIGHT, NUM_CHANNELS
from my_utils.metrics import compute_metrics
from my_utils.encoding_convertions import krnConverter


class RNNTTrainedCRNN:
    """
    Wrapper de entrenamiento para el modelo RNN-Transducer.

    Diferencias respecto a CTCTrainedCRNN:
    - Pérdida: torchaudio.transforms.RNNTLoss en lugar de CTCLoss.
    - Arquitectura: encoder + prediction network + joint network.
    - Vocabulario: requiere token <BLANK> (se reutiliza <PAD>).
    - El data generator debe proporcionar input_lengths y target_lengths reales.
    - Inferencia: decodificación greedy time-synchronous del transductor.
    """

    def __init__(
        self,
        dictionaries: Tuple[Dict[str, int], Dict[int, str]],
        encoding: str,
        device: torch.device,
    ):
        self.w2i, self.i2w = dictionaries
        self.blank_index = self.w2i["<PAD>"]    # <PAD> se reutiliza como blank
        self.device = device
        self.krnParser = krnConverter(encoding=encoding)

        self.model = RNNTModel(vocab_size=len(self.w2i))
        self.model.to(self.device)
        self.compile()
        self.summary()

    def summary(self):
        summary(self.model.encoder, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def compile(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # RNNTLoss espera logits sin log_softmax; lo aplica internamente
        self.compute_rnnt_loss = torchaudio.transforms.RNNTLoss(
            blank=self.blank_index,
            reduction="mean",
        )

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
        self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        x, xl, y, yl = data
        # x:  (B, C, H, W)
        # xl: (B,) longitudes de entrada tras reducción del CNN (eje T del encoder)
        # y:  (B, S) tokens objetivo SIN <SOS>/<EOS>, relleno con <PAD>
        # yl: (B,) longitudes reales de cada secuencia objetivo
        self.optimizer.zero_grad()

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_out = self.model.encoder(x)                 # (B, T, enc_dim)

        # ── Prediction network ───────────────────────────────────────────────
        # Anteponer blank como primer token de contexto: (B, S+1)
        B, S = y.shape
        blank_col = torch.full((B, 1), self.blank_index, dtype=torch.long, device=self.device)
        pred_input = torch.cat([blank_col, y], dim=1)   # (B, S+1)
        pred_out, _ = self.model.prediction(pred_input) # (B, S+1, pred_dim)

        # ── Joint network ────────────────────────────────────────────────────
        # logits: (B, T, S+1, vocab_size)
        logits = self.model.joint(enc_out, pred_out)

        # ── RNNTLoss ─────────────────────────────────────────────────────────
        # Requiere targets sin relleno: extraemos hasta yl[i] por muestra
        # torchaudio acepta el tensor con padding; usa yl para ignorarlo
        loss = self.compute_rnnt_loss(
            logits.float(),
            y.int(),
            xl.int(),   # longitudes del encoder (eje T)
            yl.int(),   # longitudes del objetivo (eje S)
        )
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

    def _greedy_decode(self, enc_out: torch.Tensor) -> List[str]:
        """
        Decodificación greedy time-synchronous del transductor.

        Por cada paso temporal t del encoder:
          1. Se obtiene g_u de la prediction network con el último token emitido.
          2. Se calcula el logit de la joint network en (t, u).
          3. Si argmax != blank → se emite el token y se avanza u (sin avanzar t).
          4. Si argmax == blank → se avanza t (sin emitir nada).
        """
        # enc_out: (1, T, enc_dim)
        T = enc_out.size(1)
        pred_tokens = []

        last_token = torch.tensor([[self.blank_index]], device=self.device)  # (1, 1)
        hidden = None

        for t in range(T):
            f_t = enc_out[:, t:t+1, :]          # (1, 1, enc_dim)
            while True:
                g_u, hidden = self.model.prediction(last_token, hidden)   # (1, 1, pred_dim)
                logit = self.model.joint(f_t, g_u)    # (1, 1, 1, vocab_size)
                token_idx = logit.squeeze().argmax(-1).item()

                if token_idx == self.blank_index:
                    break                             # avanzar al siguiente frame
                pred_tokens.append(self.i2w[token_idx])
                last_token = torch.tensor([[token_idx]], device=self.device)

        return pred_tokens

    def evaluate(
        self,
        XFiles: List[str],
        YFiles: List[str],
        krnParser: krnConverter,
        aux_name: str,
        print_metrics: bool = True,
        print_random_samples: bool = False,
    ) -> Dict[str, float]:
        Y = []
        YPRED = []

        with torch.no_grad():
            for xf, yf in zip(XFiles, YFiles):
                x = preprocess_audio(
                    xf,
                    training=False,
                    width_reduction=self.model.encoder.width_reduction,
                )
                x = torch.from_numpy(x.copy()).float().unsqueeze(0).to(self.device)

                enc_out = self.model.encoder(x)         # (1, T, enc_dim)
                YPRED.append(self._greedy_decode(enc_out))

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