# CharGPT
This repository contains the implementation of a GPT-like Transformer model built from scratch using PyTorch. The project is designed to train a character-level language model on the Tiny Shakespeare dataset, enabling the model to generate text by predicting the next character based on the previous ones.
# Project Overview
The primary goal of this project is to build a simplified GPT-like model using a Transformer architecture. The model is designed for language modeling, where it learns to generate text by predicting the probability of the next character in a sequence based on the prior characters.
## Model Details
The model follows a scaled-down version of the GPT architecture:

- **Embedding Layer**: Converts input tokens into dense vectors.
- **Multi-Head Self-Attention**: The model employs multiple heads to learn different aspects of attention across sequences.
- **Feed Forward Network (FFN)**: Applies non-linearity and learns deep representations.
- **Positional Encoding**: Adds positional information to input embeddings to preserve the order of the tokens.
- **Residual Connections & Layer Normalization**: To stabilize training and allow gradient flow.
## Notebook Overview
All the work, including model definition, training, and evaluation, is included in the Jupyter notebook file:

**GPT_Dev.ipynb**: This notebook contains the complete workflow:
- Data loading and preprocessing.
- Model architecture (Transformer-based GPT-like model).
- Training loop with evaluation on training and validation sets.
- Text generation using the trained model.
# How to Use the Notebook
1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/gpt-language-model.git
   cd gpt-language-model
2. **Install Dependencies**:
Make sure to install all necessary libraries using:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook: Start Jupyter Notebook and open language_model.ipynb to train the model or generate text**.
    ```bash
    jupyter notebook GPT_Dev.ipynb
    
# Training:
The training is performed within the notebook. You can configure the number of iterations (max_iters), learning rate (learning_rate), and other hyperparameters as per your requirements.
The notebook also includes functions to evaluate loss on both training and validation datasets during the training process.
Text Generation: Once the model is trained, you can generate new sequences by providing a starting token. The notebook includes a generate function that allows you to specify the number of tokens to generate.

# Dataset
The dataset used in this project is the Tiny Shakespeare dataset, which contains text written in Shakespeare's style. 
**You can download the dataset by running the following command in the notebook**:
   ```bash
   !curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

License
This project is licensed under the MIT License - see the LICENSE file for details.
