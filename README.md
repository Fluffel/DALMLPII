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

Note: The code uses wandb to report results. There is no clean solution to deactivate this behavior. If the code shall be run without wandb active, the following lines have to be deleted or commented in main.py:

        91 wandb.login()
        92 if strat == "random":
        93    config = {"initial_size": initial_size, "sampling_batch_size": sampling_batch_size, "iterations":iterations, "epochs": epochs, "learning_rate":lr, "seed": None}
        94    wandb.init(project="DiscriminativeActiveLearning", name="RandomStrategy", config=config)
        95 elif strat == "dal":
        96    config = {"initial_size": initial_size, "sampling_batch_size": sampling_batch_size, "iterations":iterations, "sub_iterations_dal":sub_iterations_dal, "epochs": epochs, "discriminator_epochs":discriminator_epochs, "learning_rate":lr, "seed": None}
        97    wandb.init(project="DiscriminativeActiveLearning", name="DiscriminativeActiveLearning", config=config)
        98 else:
        99    config = {"epochs": epochs, "learning_rate":lr, "seed": None}
       100    wandb.init(project="DiscriminativeActiveLearning", name="LeNetPlain", config=config)
        .
        .
        .
       155    wandb.log({"test/acc": acc, "test/loss": loss})
        .
        .
        .
       177    wandb.log({"test/acc": acc, "test/loss": loss})
