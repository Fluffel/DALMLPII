# Project in Machine Learning for NLP II
This project implements the discriminative active learning strategy proposed in https://arxiv.org/abs/1907.06347.
Models for training can be found in model.py. It uses LeNet-5 to fit on FashionMNIST and a multilayer perceptron as a binary classifier for the DAL strategy.
The strategy is implemented in strategy.py, as well as a random sampling strategy as baseline.

## Run the Code
To get the dependencies run:

    pip install -r requirements.txt

To run the code, execute main.py. Parameters have to be adjusted in the code itself, mainly in:

    if __name__ == '__main__':
      main("Random", initial_size=100, strat='random', sampling_batch_size=100, iterations=20, sub_iterations_dal=1, epochs=140, discriminator_epochs=700, lr=1e-4, seed=None, no_gpu=True)
