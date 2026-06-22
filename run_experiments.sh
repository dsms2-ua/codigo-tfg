#!/bin/bash

metodo="rnn-t" #Valores posibles: ctc, rnn-t, seq2seq
encoding="decoupled" #Valores posibles: decoupled, decoupled-dot, kern

for i in {1..6}; do
    python3 train.py --method $metodo --experiment_id $i --encoding $encoding --batch_size 16 --epochs 300 --patience 25
done

# Experimentos que faltan (1 al 6):
# RNN_T => decoupled / decoupled-dot / kern
# Seq2Seq => decoupled / decoupled-dot / kern
