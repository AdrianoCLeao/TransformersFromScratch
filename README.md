# Transformer from Scratch with PyTorch

## Overview

This project aims to implement a Transformer model from scratch using PyTorch, inspired by the seminal paper "Attention is All You Need" by Vaswani et al. The Transformer architecture introduced in this paper revolutionized natural language processing and machine translation by relying solely on attention mechanisms, discarding recurrence and convolutions.

## Project Structure

- `model.py`: Contains the implementation of the Transformer model, including the Encoder, Decoder, and the overall Transformer architecture.
- `utils.py`: Includes utility functions for data processing, tokenization, and training.
- `train.py`: Script to train the Transformer model on a dataset.
- `evaluate.py`: Script to evaluate the model's performance on a test set.
- `data/`: Directory containing datasets for training and evaluation.
- `notebooks/`: Jupyter notebooks with exploratory data analysis and model training experiments.
- `requirements.txt`: List of required Python packages for the project.

## Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
