import numpy as np

def get_unlabeled_idc(train_size, labeled_idc):
    return np.setdiff1d(np.arange(train_size), labeled_idc)


class SamplingStrategy():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # self.seed = seed


class DiscrimativeRepresentationSampling(SamplingStrategy):
    def __init__(self, model, gpu):
        self.model = model
        self.gpu = gpu

    def next_sample(self, X_train, labeled_idc):
        pass

class RandomSampling(SamplingStrategy):
    def __init__(self, batch_size):
        super().__init__(batch_size)
    def next_sample(self, sample_set, labeled_idc):
        # np.random.seed(self.seed)
        unlabeled_idc = get_unlabeled_idc(len(sample_set), labeled_idc)
        return np.hstack((np.random.choice(unlabeled_idc, self.batch_size), labeled_idc))
