# Transformer from Scratch with PyTorch
<img src="https://github.com/user-attachments/assets/f5e88463-9f28-44b9-8dbc-240da904e859" alt="transformer" width="450">

## Overview
This project implements an English-to-Portuguese translation system using a Transformer model built from scratch. The model was developed based on the paper "Attention Is All You Need" (Vaswani et al., 2017), which introduced the Transformer architecture for translation and other natural language processing tasks.

## Project Description

The goal of this project is to build a machine translation model that can accurately translate English texts into Portuguese. The model was implemented from scratch without using pre-built Transformer libraries and is trained using the **Helsinki-NLP/opus_books** dataset available on Hugging Face Datasets.

## Dataset

The dataset used is [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-pt), which contains a collection of books translated into English and Portuguese. This dataset is ideal for training translation models as it provides sentence pairs in both languages.

## Model Architecture
The Transformer model consists of:

- Encoder: Encodes the input sequence into an internal representation.
- Decoder: Decodes the internal representation to generate the output sequence.
- Attention Layers: Used to capture relationships between different parts of the input and output.
  
Training is performed using cross-entropy loss, and the model is optimized with the Adam optimizer.

## Usage

1. Train the Model
Run the train.py script to train the model:

```bash
python .\train\train.py
```
The script will load the dataset, train the Transformer model, and save the model weights after each epoch.

2. Translate Text
After training, you can use the translate.py script to translate English text into Portuguese:

```bash
python translate.py "Your English text here"
```
If you do not provide text, the script will use a default example.
