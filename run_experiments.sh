#!/bin/bash

metodo="ctc" #Valores posibles: ctc, rnn-t, seq2seq

for i in {1..6}; do
    python3 train.py --method $metodo --experiment_id $i --encoding decoupled --batch_size 16 --epochs 300 --patience 25
done
