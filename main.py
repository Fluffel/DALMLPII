import os
import torch
from torch.utils.data import Subset

from model import *
from strategies import *
from utils import *

import timeit


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





def train_model(model_path, train_set, device, lr=1e-3, epochs=200, early_stop=False):
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.train()

    train_loader = get_data_loader(train_set)
    optimizer = get_SGD_optimizer(model, lr)
    # optimizer = get_adam_optimizer(model, lr)
    loss_function = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train_step(model, train_loader, optimizer, loss_function, device)

    return model



def main(initial_size, seed, filename, strat='dal', sampling_batch_size=100, iterations=1, sub_iterations_dal=4, discriminator_epochs=550, no_gpu=True):
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
    
    print("Load Dataset...")
    train_set, test_set = load_mnist()
    test_loader = get_data_loader(test_set)
    labeled = np.random.permutation(len(train_set))[:initial_size]

    if strat == 'random':
        strategy = RandomSampling(sampling_batch_size)
    elif strat == 'dal':
        strategy = DiscrimativeRepresentationSampling(sampling_batch_size, None, num_sub_batches=sub_iterations_dal, discriminator_epochs=discriminator_epochs, device=device)

    print("-----------Start Iterations-----------")
    accuracies = []
    losses = []
    number_labeled = []
    current_iteration = 0
    print("Train LeNet...")
    current_model = train_model(model_path, Subset(train_set, labeled), device)
    loss, acc = eval_model(current_model, test_loader, nn.CrossEntropyLoss(), device)
    losses.append(loss)
    accuracies.append(acc)
    number_labeled.append(len(labeled))
    print("-----------Iteration "+ str(current_iteration) + "-----------")
    print("Labeled sample size: " + str(len(labeled)))
    print("Accuracy: " + str(acc))
    print("Loss: " + str(loss))
    for i in range(iterations):
        print("Update model...")
        strategy.update_model(current_model)
        labeled = strategy.next_sample(train_set, labeled)
        labeled_dataset = Subset(train_set, labeled)

        print("Train LeNet...")
        current_model = train_model(model_path, labeled_dataset, device)
        loss, acc = eval_model(current_model, test_loader, nn.CrossEntropyLoss(), device)
        losses.append(loss)
        accuracies.append(acc)
        number_labeled.append(len(labeled))
        current_iteration += 1
        print("-----------Iteration "+ str(current_iteration) + "-----------")
        print("Labeled sample size: " + str(len(labeled)))
        print("Accuracy: " + str(acc))
        print("Loss: " + str(loss))
    
    save_result({"accuracy": accuracies, "losses": losses, "number labeled": number_labeled}, filename)


if __name__ == '__main__':
    start = timeit.timeit()
    main(100, None, "data/acc_loss_rand_sgd.json", strat='random', sampling_batch_size=100, iterations=20)
    end = timeit.timeit()
    print("total elapsed time: {}".format(end - start))

    