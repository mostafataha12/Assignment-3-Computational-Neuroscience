RNN Text Prediction Assignment
This repository contains a PyTorch implementation of a Recurrent Neural Network (RNN) designed to predict the fourth word in a four-word text sequence. The project is part of an exercise to build an RNN model for a custom text dataset, inspired by a provided MNIST RNN example.
Project Overview
The goal of this assignment is to create an RNN that takes the first three words of a four-word phrase and predicts the fourth word. The dataset consists of five simple phrases, and the model uses word embeddings and an RNN to learn the sequence patterns.
Dataset
The dataset includes the following phrases:

"the cat is fluffy"
"dog runs very fast"
"bird flies so high"
"fish swims in water"
"ant works all day"

Each phrase is processed to use the first three words as input and the fourth word as the target. The vocabulary contains 20 unique words, and the model predicts the target word from this vocabulary.
Model

Architecture: The model consists of an embedding layer, a multi-layer RNN, and a fully connected layer.
Hyperparameters:
Input size: 10 (embedding dimension)
Hidden size: 128
Number of layers: 2
Number of classes: 20 (equal to vocabulary size)
Sequence length: 3 (first three words)
Epochs: 200
Batch size: 4
Learning rate: 0.001


Loss Function: CrossEntropyLoss
Optimizer: Adam

Prerequisites
To run the code, you need the following:

Python 3.8 or higher
PyTorch 1.9 or higher
NumPy

Optional:

CUDA-compatible GPU (for faster training, if available)

Installation

Clone the Repository:
git clone https://github.com/your-username/rnn-text-prediction.git
cd rnn-text-prediction


Create a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install torch numpy

If you have a GPU and want to use CUDA, ensure you install the appropriate PyTorch version:
pip install torch --index-url https://download.pytorch.org/whl/cu118



Usage

Run the Script:The main script is text_rnn_fixed.py. To train the model and test it, run:
python text_rnn_fixed.py

