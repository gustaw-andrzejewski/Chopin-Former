# ChopinFormer

A GPT-style transformer implementation for MIDI music generation, built from scratch with PyTorch and PyTorch Lightning.

## Overview

ChopinFormer is a transformer-based language model designed to generate piano music in the style of classical composers. This project serves as a learning exercise in transformer architecture implementation while exploring the interesting application of language models to music generation.

## Features

- Built on a decoder-only transformer architecture similar to GPT
- Uses relative attention to better capture music's temporal relationships
- Works with MIDI data tokenized using MidiTok
- Trained on the MAESTRO dataset of piano performances

## Why Relative Attention?

Music has inherent patterns that depend on relative positioning of notes rather than absolute positions. Relative attention allows the model to understand musical patterns like phrases and motifs regardless of where they appear in a piece, which is particularly important for capturing musical structure.

## Project Structure

- `model/` - Core transformer implementation 
- `dataset/` - Data processing for MIDI files
- `train.py` - Training script

## Requirements

- PyTorch
- PyTorch Lightning
- MidiTok
- wandb (for logging)

## Usage

Train the model:

```bash
python train.py --midi-dir ./data/maestro-v3.0.0 --augment
```

## Purpose

This project was created to:

1. Learn transformer architecture by implementing it from scratch
2. Explore applying transformers to the domain of music generation
3. Understand the benefits of relative attention in sequential musical data
