import gc
import math

import numpy as np
from model import *
import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset
from utils import *
import wandb


class SamplingStrategy():
    def __init__(self, batch_size, iteration=0):
        self.batch_size = batch_size
        self.current_iteration = iteration
        # self.seed = seed
    def update_model(self, model) -> None:
        raise NotImplementedError()

    def get_next_batch(self, sample_set, labeled_idc):
        raise NotImplementedError()
    def next_iteration(self):
        self.current_iteration += 1


class RandomSampling(SamplingStrategy):
    def __init__(self, batch_size):
        super().__init__(batch_size)
    def get_next_batch(self, sample_set, labeled_idc):
        # np.random.seed(self.seed)
        unlabeled_idc = get_unlabeled_idc(len(sample_set), labeled_idc)
        return np.hstack((np.random.choice(unlabeled_idc, self.batch_size), labeled_idc))
    
    def update_model(self, model) -> None:
        return None


class BinaryData(Dataset):
    def __init__(self, data, classes) -> None:
        self.data = data
        self.classes = classes

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data[index], self.classes[index])

class ModelNotAvailable(Exception):
    def __init__(self, message="Embedding model is None") -> None:
        self.message = message
        super().__init__(self.message)
class DiscrimativeRepresentationSampling(SamplingStrategy):
    def __init__(self, batch_size, model, num_sub_batches, discriminator_epochs, device):
        super().__init__(batch_size)
        if model == None:
            self.model = model
        else:
            self.model = LeNetEmbeddingModel(model).to(device)
        self.discriminator_epochs = discriminator_epochs
        self.device = device
        self.sub_batch_size = int(math.ceil(batch_size / num_sub_batches))
        self.current_subiteration = 0
        self.num_sub_batches = num_sub_batches 


    def get_embeddings(self, data):
        self.model.eval()
        embeddings = torch.Tensor()
        with torch.no_grad():
            for x, _ in data:
                x = x.to(self.device)
                output = self.model(x)
                embeddings = torch.cat((embeddings, output), dim=0)

        return embeddings
    
    def train_discriminator(self, data, unlabeled_idc, labeled_idc):
        model = DiscriminativeMultilayerPerceptron(data.shape[1]).to(self.device)
        
        model.train()

        classes = np.zeros(data.shape[0])
        classes[labeled_idc] = 1
        all_idc = np.hstack((labeled_idc, unlabeled_idc))
        train_x = data[all_idc]
        train_y = classes[all_idc]

        loader = get_data_loader(BinaryData(train_x, torch.tensor(train_y, dtype=torch.long, device=self.device)))
        # optimizer = get_SGD_optimizer(model, lr=0.0001, momentum=0.1)
        optimizer = get_adam_optimizer(model, lr=0.0001)
        early_stopper = EarlyStop(40, 0.0001)
        
        weights = torch.tensor([train_x.shape[0] / (len(unlabeled_idc)), train_x.shape[0] / (len(labeled_idc))], device=self.device)
        loss_function = nn.CrossEntropyLoss(weight=weights) 

        print_count = 0 
        print("---------- Train Discriminator ---------- ")
        for e in range(self.discriminator_epochs):
            train_step(model, loader, optimizer, loss_function, self.device)

            l, acc = eval_model(model, loader, loss_function, self.device)
            # step_idx = (self.current_iteration - 1) * self.num_sub_batches + self.current_subiteration
            # wandb.log({"discriminator/accuracy": acc, 
            #            " discriminator/loss": l}, step=step_idx, name="Iter {}-{}".format(self.current_iteration, self.current_subiteration))
            # wandb.log({"discriminator/accuracy {}-{}".format(self.current_iteration, self.current_subiteration): acc, 
            #            "discriminator/loss {}-{}".format(self.current_iteration, self.current_subiteration): l}, step=e)
            # prefix = "discriminator/{}-{}".format(self.current_iteration, self.current_subiteration)
            # wandb.define_metric(prefix)
            # wandb.define_metric("discriminator/*", step_metric=prefix)
            # wandb.log({prefix + "/accuracy": acc, 
            #            prefix + "/loss": l})
            if print_count % 40 == 0: #print every fourtieth iteration
                print("---- Epoch " + str(e) + " ----")
                print("Acc: {}".format(acc))
                print("Loss: {}".format(l))
            if early_stopper.early_stop(l):
                break
            print_count += 1
        return model




    def get_next_batch(self, sample_set, labeled_idc):
        def get_unlabeled_idc_with_ratio(set_size, labeled_idc, ratio):
            unlabeled_idc = get_unlabeled_idc(set_size, labeled_idc)
            if len(labeled_idc) * ratio < len(unlabeled_idc):
                unlabeled_idc = np.random.choice(unlabeled_idc, ratio * len(labeled_idc), replace=False)

            return unlabeled_idc
        

        if self.model == None:
            raise ModelNotAvailable()
        
        unlabeled_labeled_max_ratio = 10
        sample_loader = DataLoader(sample_set, batch_size=64, shuffle=False)
        print("Retrieve embeddings...")
        embeddings = self.get_embeddings(sample_loader)

        sample_count = 0
        self.current_subiteration = 0
        print("--- Train Discriminator ---")
        while sample_count < self.batch_size:
            print("Subbatch {}".format(self.current_subiteration))
            unlabeled_idc = get_unlabeled_idc_with_ratio(len(sample_set), labeled_idc, unlabeled_labeled_max_ratio)
            sub_batch_size = min(self.sub_batch_size, self.batch_size - sample_count)
            discriminator = self.train_discriminator(embeddings, unlabeled_idc, labeled_idc)
            discriminator.eval()
            with torch.no_grad():
                predictions = discriminator(embeddings[unlabeled_idc])

                to_labeled_idc = np.argpartition(predictions[:,0].cpu(), -sub_batch_size)[-sub_batch_size:]
                labeled_idc = np.hstack((labeled_idc, to_labeled_idc))
                sample_count += sub_batch_size
                
            self.current_subiteration += 1
            del discriminator
            gc.collect()

        return labeled_idc




    def update_model(self, model) -> None:
        del self.model
        gc.collect()
        self.model = LeNetEmbeddingModel(model).to(self.device)
        