#!/bin/bash

set -e

# Créer le dossier de sortie si nécessaire
mkdir -p generation/features

# Paramètres
WAV_FILE="data/wav/input.wav"
RAW_AUDIO="generation/features/audio.raw"
MFCC_OUT="generation/features/mfcc.raw"
SAMPFREQ=16000   # échantillonnage
FRAMELEN=512     # taille fenêtre (≈32 ms à 16 kHz)
FRAMEPERIOD=256  # hop-size (≈16 ms)
NUM_MFCC=12      # nombre de coefficients MFCC

# Convertir WAV en float
sox "$WAV_FILE" -t raw -r $SAMPFREQ -b 16 -c 1 -e signed-integer "$RAW_AUDIO"

# Pipeline SPTK
cat "$RAW_AUDIO" | \
x2x +sf | \
frame -l $FRAMELEN -p $FRAMEPERIOD | \
window -l $FRAMELEN -L $FRAMELEN -w 1 | \
mfcc -l $FRAMELEN -m $NUM_MFCC > "$MFCC_OUT"

echo "MFCC written to $MFCC_OUT"
