# CharGPT
This repository contains the implementation of a GPT-like Transformer model built from scratch using PyTorch. The project is designed to train a character-level language model on tinyshakespeare dataset, which can generate text by predicting the next token based on previous ones.
# Project Overview
The primary goal of this project is to build a simplified GPT-like model using a Transformer architecture. The model is designed for language modeling tasks, where it learns to generate text by predicting the probability of the next word in a sequence based on previous words.
# Notebook Overview
All the work including model definition, training, and evaluation is included in the Jupyter notebook file:

language_model.ipynb: This notebook contains the complete workflow:
Data loading and preprocessing.
Model architecture (Transformer-based GPT-like model).
Training loop with evaluation on training and validation sets.
Text generation using the trained model.
# Model Details
The model follows a scaled-down version of the GPT architecture:

Embedding Layer: Converts input tokens into dense vectors.
Multi-Head Self-Attention: The model employs multiple heads to learn different aspects of attention across sequences.
Feed Forward Network (FFN): Applies non-linearity and learns deep representations.
Positional Encoding: Adds positional information to input embeddings to preserve the order of the tokens.
Residual Connections & Layer Normalization: To stabilize training and allow gradient flow.
#
