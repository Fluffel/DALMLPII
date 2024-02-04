import os
import torch
from torch.utils.data import Subset, random_split

from model import *
from strategies import *
from utils import *

import wandb


def load_mnist(batch_size = 64, normalize=True):
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor, Normalize, Compose


    path_to_mean_std = "data/mnist_mean_std.pt"
    if normalize and os.path.exists(path_to_mean_std):
        loaded_tensors = torch.load(path_to_mean_std)
        train_mean = loaded_tensors['train_mean']
        train_std = loaded_tensors['train_std']
        test_mean = loaded_tensors['test_mean']
        test_std = loaded_tensors['test_std']

        transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
        train_set = FashionMNIST('./data', train=True, download=True, transform=transform)

        transform = Compose([ToTensor(), Normalize(test_mean, test_std)])
        test_set = FashionMNIST('./data', train=False, download=True, transform=transform)

        return train_set, test_set
    
    transform = ToTensor()
    train_set = FashionMNIST('./data', train=True, download=True, transform=transform)
    test_set = FashionMNIST('./data', train=False, download=True, transform=transform)

    # Normalize
    if normalize:
        train_mean, train_std = get_mean_std(train_set)
        test_mean, test_std = get_mean_std(test_set)
        
        torch.save({'train_mean': train_mean,
                    'train_std': train_std,
                    'test_mean': test_mean,
                    'test_std': test_std}, path_to_mean_std)

        transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
        train_set = FashionMNIST('./data', train=True, download=True, transform=transform)

        transform = Compose([ToTensor(), Normalize(test_mean, test_std)])
        test_set = FashionMNIST('./data', train=False, download=True, transform=transform)
    
    return train_set, test_set





def train_model(model_path, train_set, validation_set, iteration, device, lr=1e-3, epochs=200, early_stop=False):
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.train()

    train_loader = get_data_loader(train_set)
    validation_loader = get_data_loader(validation_set)
    # optimizer = get_SGD_optimizer(model, lr)
    optimizer = get_adam_optimizer(model, lr)
    loss_function = nn.CrossEntropyLoss()
    if early_stop:
        early_stopper = EarlyStop(15, 0.0001)

    epoch_count = 0
    # print("Eval...")
    # loss, acc = eval_model(model, validation_loader, loss_function, device)
    # wandb.log({"validation/accuracy": acc, validation"/loss": loss})
    for e in range(epochs):
        # print("Train...")
        train_step(model, train_loader, optimizer, loss_function, device)
        # print("Eval...")
        # wandb.log({"validation/accuracy": acc, "validation/loss": loss})
        if early_stop:
            loss, acc = eval_model(model, validation_loader, loss_function, device)
            if early_stopper.early_stop(loss):
                break
        epoch_count += 1
    return model



def main(initial_size, strat, sampling_batch_size=100, iterations=1, sub_iterations_dal=4, epochs=200, discriminator_epochs=550, lr=1e-3, train_size=100, seed=None, no_gpu=True):
    wandb.login()
    if strat == "random":
        config = {"initial_size": initial_size, "sampling_batch_size": sampling_batch_size, "iterations":iterations, "epochs": epochs, "learning_rate":lr, "seed": None}
        wandb.init(project="DiscriminativeActiveLearning", name="RandomStrategy", config=config)
    elif strat == "dal":
        config = {"initial_size": initial_size, "sampling_batch_size": sampling_batch_size, "iterations":iterations, "sub_iterations_dal":sub_iterations_dal, "epochs": epochs, "discriminator_epochs":discriminator_epochs, "learning_rate":lr, "seed": None}
        wandb.init(project="DiscriminativeActiveLearning", name="DiscriminativeActiveLearning", config=config)
    else:
        config = {"epochs": epochs, "learning_rate":lr, "seed": None}
        wandb.init(project="DiscriminativeActiveLearning", name="LeNetPlain", config=config)


    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

    use_gpu = not no_gpu and torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Initialize model...")
    model_path = 'data/initial_model'
    model = LeNet().to(device)
    torch.save(model.state_dict(), model_path)
    
    # Load Dataset
    print("Load Dataset...")
    train_set, test_set = load_mnist()
    train_set_size = int(0.8 * len(train_set))
    test_loader = get_data_loader(test_set)
    train_set, validation_set = random_split(train_set, [train_set_size, len(train_set) - train_set_size])
    labeled = np.random.permutation(len(train_set))[:initial_size]

    # Initialize strategy
    if strat == 'random':
        strategy = RandomSampling(sampling_batch_size)
    elif strat == 'dal':
        strategy = DiscrimativeRepresentationSampling(sampling_batch_size, None, num_sub_batches=sub_iterations_dal, discriminator_epochs=discriminator_epochs, device=device)
    else:
        # If no strategy is selected, the model is just trained on a custom sized train set, reporting validation scores

        small_train_set, small_validation_set, _ = random_split(train_set, [train_size, int(0.5*train_size), int(len(train_set) - (1.5 * train_size))])
        current_model = train_model(model_path, small_train_set, small_validation_set, "Full Dataset", device, lr=lr, epochs=epochs, early_stop=True)
        loss, acc = eval_model(current_model, test_loader, nn.CrossEntropyLoss(), device)
        print("Test loss: {}".format(loss))
        print("Test acc: {}".format(acc))
        return
    


    print("-----------Start Iterations-----------")
    accuracies = []
    losses = []
    number_labeled = []

    print("Train LeNet...")
    current_model = train_model(model_path, Subset(train_set, labeled), validation_set, 0, device, lr=lr, epochs=epochs)
    loss, acc = eval_model(current_model, test_loader, nn.CrossEntropyLoss(), device)
    losses.append(loss)
    accuracies.append(acc)
    number_labeled.append(len(labeled))

    wandb.log({"test/acc": acc, "test/loss": loss})
    print("-----------Iteration "+ str(0) + "-----------")
    print("Labeled sample size: " + str(len(labeled)))
    print("Accuracy: " + str(acc))
    print("Loss: " + str(loss))
    for i in range(iterations):
        # configure strat
        strategy.update_model(current_model)

        # Update labeled data
        labeled = strategy.get_next_batch(train_set, labeled)
        labeled_train_set = Subset(train_set, labeled)
        number_labeled.append(len(labeled))

        # Train model
        print("Train LeNet...")
        current_model = train_model(model_path, labeled_train_set, validation_set, i+1, device, lr=lr, epochs=epochs)

        # Evaluate current iteration
        loss, acc = eval_model(current_model, test_loader, nn.CrossEntropyLoss(), device)
        losses.append(loss)
        accuracies.append(acc)
        wandb.log({"test/acc": acc, "test/loss": loss})
        print("-----------Iteration "+ str(i + 1) + "-----------")
        print("Labeled sample size: " + str(len(labeled)))
        print("Accuracy: " + str(acc))
        print("Loss: " + str(loss))
    
        strategy.next_iteration()


if __name__ == '__main__':
    main(initial_size=100, strat='dal', sampling_batch_size=100, iterations=20, sub_iterations_dal=4, epochs=200, discriminator_epochs=700, lr=1e-4, seed=None, no_gpu=True)
    