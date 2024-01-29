import os
from model import *
from strategies import *
import torch
from torch.utils.data import DataLoader, Subset

def get_SGD_optimizer(model, lr, weight_decay=1e-5, momentum=0.9):
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

def get_mean_std(data):

    total_mean = torch.zeros(1)
    total_std = torch.zeros(1)
    total_samples = 0
    for images, _  in data:
        num_samples = images.shape[0]
        total_samples += num_samples
        total_mean += images.mean() * num_samples
        total_std += images.std() * num_samples

    mean = total_mean / total_samples
    std = total_std / total_samples
    return mean, std



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

def get_data_loader(data_set, batch_size=32):
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def train_step(model, loader, optimizer, loss_function, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_function(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def train_model(model_path, train_set, device, lr=1e-3, epochs=200, early_stop=False): #TODO: early stop?
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path))
    train_loader = get_data_loader(train_set)
    optimizer = get_SGD_optimizer(model, lr)
    loss_function = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train_step(model, train_loader, optimizer, loss_function, device)

    return model

def test_model(model, test_set, device):
    model.eval()
    cum_accuracy = 0
    total_loss = 0

    loss_function = nn.CrossEntropyLoss()
    test_loader = get_data_loader(test_set)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_function(output, y)
            total_loss += loss.item()  #LOOK WHAT THIS DOES
            _, predicted = output.max(dim=1)
            cum_accuracy += sum(predicted.eq(y)).item()
    return total_loss/len(test_set), cum_accuracy/len(test_set)

def main(initial_size, seed, sampling_batch_size=10, batch_size=32, iterations=20, no_gpu=True):
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_gpu = not no_gpu and torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Initialize model...")
    model_path = 'initial_model'
    model = LeNet().to(device)
    torch.save(model.state_dict(), model_path)
    
    print("Load Dataset...")
    train_set, test_set = load_mnist()
    # train_loader = DataLoader(train_set, batch_size=batch_size)
    # test_loader = DataLoader(test_set, batch_size=batch_size)
    labeled = np.random.permutation(len(train_set))[:initial_size]

    print("-----------Start Iterations-----------")
    accuracies = []
    losses = []
    current_iteration = 1

    strategy = RandomSampling(sampling_batch_size)
    current_model = train_model(model_path, Subset(train_set, labeled), device)
    loss, acc = test_model(current_model, test_set, device)
    losses.append(loss)
    accuracies.append(acc)
    print("-----------Iteration "+ str(current_iteration) + "-----------")
    print("Labeled sample size: " + str(len(labeled)))
    print("Accuracy: " + str(acc))
    print("Loss: " + str(loss))
    for i in range(iterations):

        labeled = strategy.next_sample(train_set, labeled)
        labeled_dataset = Subset(train_set, labeled)
        current_model = train_model(model_path, labeled_dataset, device)
        loss, acc = test_model(current_model, test_set, device)
        losses.append(loss)
        accuracies.append(acc)
        current_iteration += 1
        print("-----------Iteration "+ str(current_iteration) + "-----------")
        print("Labeled sample size: " + str(len(labeled)))
        print("Accuracy: " + str(acc))
        print("Loss: " + str(loss))


if __name__ == '__main__':
    main(50, 1)

    